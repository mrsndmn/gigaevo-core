"""Vocabulary transplant onto a frozen pretrained body.

Swap a pretrained LM's tokenizer for one of this study's arms, keep every
transformer weight frozen, and train only the new embedding table and output
heads. The question this answers that the 33M study could not: does the
case-factorization result survive when the body is a real 1B model, and can it
be *retrofitted* rather than pretrained?

Design notes
------------
**Gradients still traverse the whole body.** The input embedding is at layer 0,
so reaching it requires full activation backprop — this is ~2/3 of full-training
FLOPs, not a cheap head-only finetune. What is saved is optimizer state and
weight-gradient computation for 1.7B parameters.

**Chunked, checkpointed cross-entropy.** The host's native head is
2048 x 151,936; materialising float32 logits for a whole micro-batch is several
GB and autograd would retain them. ``_ce_sum`` slices the batch and recomputes
each slice's logits in backward instead.

**Init strategies.** ``random`` is the honest floor. ``mean_of_pieces`` sets each
new token's embedding to the mean of the host's rows for the pieces its surface
string tokenizes into — the standard transplant heuristic — and is what makes a
zero-training ("on the fly") measurement meaningful. Feature heads that have no
counterpart in the host (the role head) are zero-initialised so they start as a
no-op and train from there.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from experiments.vocab_factorization.tokenization import piece_string

# Above this vocabulary size, build the loss in slices rather than one tensor.
_CHUNK_VOCAB_THRESHOLD = 32000
_CE_CHUNK = 4096


def _ce_sum(
    h: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Summed cross-entropy of ``h @ weight.T`` against ``target``.

    For a large vocabulary the logits are produced in slices under
    ``checkpoint``, so only one slice exists at a time and backward recomputes
    the rest. For a small vocabulary this is a plain matmul.

    ``mask`` is a keep-mask over the vocabulary (see ``active_vocab``). Masked
    logits are driven to the float32 floor rather than ``-inf``: a row whose
    every entry is masked then degrades to a uniform distribution instead of
    NaN, so a mistake shows up as an absurd loss that can be traced rather than
    as a silent NaN halfway through a run. Masked rows receive exactly zero
    gradient through the softmax, which is what makes the mask *also* the
    "inactive tokens are not optimized" half of the convention.
    """
    V = weight.shape[0]
    floor = torch.finfo(torch.float32).min

    def block(hh, tt):
        lg = F.linear(hh, weight).float()
        if mask is not None:
            lg = lg.masked_fill(~mask, floor)
        return F.cross_entropy(lg, tt, reduction="sum")

    if V < _CHUNK_VOCAB_THRESHOLD:
        return block(h, target)

    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], _CE_CHUNK):
        total = total + torch.utils.checkpoint.checkpoint(
            block, h[i : i + _CE_CHUNK], target[i : i + _CE_CHUNK], use_reentrant=False
        )
    return total


def _ce_tokens(
    h: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-position cross-entropy, same arithmetic as ``_ce_sum`` unreduced.

    A corpus-level BPB only ever needs the sum, so ``_ce_sum`` reduces inside
    the chunk loop and never materializes the per-token vector. A multiple-
    choice benchmark needs the likelihood of one *sequence* at a time, so the
    reduction has to stay per-position. Slicing here is purely about the size
    of the fp32 logits (a 151,936-wide row per token); there is no backward, so
    unlike ``_ce_sum`` this does not need ``checkpoint`` to recompute them.

    ``ignore_index=-100`` carries padding through: batching choices of
    different lengths is unavoidable here, and a padded position must
    contribute exactly zero rather than a small wrong number.
    """
    floor = torch.finfo(torch.float32).min
    parts = []
    for i in range(0, h.shape[0], _CE_CHUNK):
        lg = F.linear(h[i : i + _CE_CHUNK], weight).float()
        if mask is not None:
            lg = lg.masked_fill(~mask, floor)
        parts.append(
            F.cross_entropy(
                lg, target[i : i + _CE_CHUNK], reduction="none", ignore_index=-100
            )
        )
    return torch.cat(parts)


class TransplantLM(nn.Module):
    """Frozen pretrained body + trainable factorized embedding and heads.

    Exposes the same ``forward(inputs, targets) -> {head: summed NLL}`` contract
    as ``model.FactorizedLM``, so the bits-per-byte evaluator is shared verbatim
    between the two studies.
    """

    def __init__(
        self,
        body: nn.Module,
        heads: dict[str, int],
        d_model: int,
        tie_embeddings: bool = True,
        freeze_body: bool = True,
    ):
        super().__init__()
        self.body = body
        self.heads = dict(heads)
        self.head_names = list(heads)
        self.d_model = d_model
        self.freeze_body = freeze_body
        # Freezing is the study's default and the thing every published number
        # here assumes: it is what makes "retrofit a vocabulary onto a model you
        # did not train" a meaningful claim. Unfreezing is one deliberate arm,
        # asking whether the residual BPB gap is the embedding's fault or the
        # body's refusal to move — and it changes the memory profile completely
        # (optimizer state for 1.7B rather than 0.3B parameters), so it carries
        # its own micro-batch in ``configs.TRANSPLANTS``.
        for p in self.body.parameters():
            p.requires_grad_(not freeze_body)
        self._body_dtype = next(self.body.parameters()).dtype

        self.emb = nn.ModuleDict(
            {name: nn.Embedding(card, d_model) for name, card in heads.items()}
        )
        self.out = nn.ModuleDict(
            {name: nn.Linear(d_model, card, bias=False) for name, card in heads.items()}
        )
        if tie_embeddings:
            self.out["stem"].weight = self.emb["stem"].weight
        # Feature heads have no counterpart in the host: start as a no-op.
        for name in self.head_names:
            if name != "stem":
                nn.init.zeros_(self.emb[name].weight)
                nn.init.zeros_(self.out[name].weight)
        self.register_buffer("stem_mask", None, persistent=False)

    def set_active_mask(self, mask: torch.Tensor | None) -> None:
        """Restrict the stem softmax to the active vocabulary (or lift it).

        Only the stem head is masked. The feature heads (role, suffix) have a
        few hundred values and the corpus exercises all of them, so there is
        nothing inactive to remove and a mask there would only be a chance to
        get the bookkeeping wrong.
        """
        if mask is None:
            self.stem_mask = None
            return
        if mask.shape != (self.heads["stem"],):
            raise ValueError(
                f"active mask is {tuple(mask.shape)}, expected "
                f"({self.heads['stem']},) — the mask must cover the *table*, "
                "which for an HF host is the padded row count, not len(tokenizer)"
            )
        self.stem_mask = mask.to(
            device=self.emb["stem"].weight.device, dtype=torch.bool
        )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, inputs: dict[str, torch.Tensor], targets=None):
        x = self.emb["stem"](inputs["stem"])
        for name in self.head_names:
            if name != "stem":
                x = x + self.emb[name](inputs[name])
        # Trainable tables are fp32 for stable AdamW; the frozen body is bf16.
        x = x.to(self._body_dtype)
        h = self.body(inputs_embeds=x).last_hidden_state
        if targets is None:
            out = {n: self.out[n](h) for n in self.head_names}
            if self.stem_mask is not None:
                out["stem"] = out["stem"].masked_fill(
                    ~self.stem_mask, torch.finfo(torch.float32).min
                )
            return out

        hf = h.reshape(-1, self.d_model)
        losses = {
            "stem": _ce_sum(
                hf,
                self.out["stem"].weight,
                targets["stem"].reshape(-1),
                mask=self.stem_mask,
            )
        }
        if len(self.head_names) > 1:
            # Same teacher-forced conditioning as the from-scratch study, so the
            # two sets of numbers mean the same thing.
            # clamp() keeps masked targets (-100) in range; cross_entropy's
            # ignore_index discards their loss.
            hc = (h + self.emb["stem"](targets["stem"].clamp(min=0))).reshape(
                -1, self.d_model
            )
            for name in self.head_names:
                if name != "stem":
                    losses[name] = _ce_sum(
                        hc, self.out[name].weight, targets[name].reshape(-1)
                    )
        return losses

    @torch.no_grad()
    def token_nll(self, inputs: dict, targets: dict) -> dict[str, torch.Tensor]:
        """Per-position NLL ``[B, T]`` per head, conditioned exactly as ``forward``.

        The feature heads are scored on ``h + emb_stem(target_stem)``, the same
        teacher-forced conditioning the training loss and the BPB evaluator use.
        Scoring them on ``h`` alone — which is what the ``targets=None`` branch
        returns, since that path exists to hand back raw logits — would quietly
        change what a factorized arm's likelihood *means* and make it
        incomparable with its own BPB.
        """
        x = self.emb["stem"](inputs["stem"])
        for name in self.head_names:
            if name != "stem":
                x = x + self.emb[name](inputs[name])
        h = self.body(inputs_embeds=x.to(self._body_dtype)).last_hidden_state
        B, T, _ = h.shape

        out = {
            "stem": _ce_tokens(
                h.reshape(-1, self.d_model),
                self.out["stem"].weight,
                targets["stem"].reshape(-1),
                mask=self.stem_mask,
            ).view(B, T)
        }
        if len(self.head_names) > 1:
            hc = (h + self.emb["stem"](targets["stem"].clamp(min=0))).reshape(
                -1, self.d_model
            )
            for name in self.head_names:
                if name != "stem":
                    out[name] = _ce_tokens(
                        hc, self.out[name].weight, targets[name].reshape(-1)
                    ).view(B, T)
        return out

    def param_counts(self) -> dict[str, int]:
        emb = sum(p.numel() for p in self.emb.parameters())
        out = sum(
            p.numel()
            for n, m in self.out.items()
            if not (n == "stem" and m.weight is self.emb["stem"].weight)
            for p in m.parameters()
        )
        # ``self.body`` still owns the host's embed_tokens, but we feed it
        # inputs_embeds and never use that table. Counting it would inflate
        # every arm by the very 311M parameters the transplant replaces.
        host_emb = self.body.get_input_embeddings()
        bypassed = sum(p.numel() for p in host_emb.parameters()) if host_emb else 0
        body = sum(p.numel() for p in self.body.parameters()) - bypassed
        return {
            "embedding": emb,
            "output_heads": out,
            "ngram": 0,
            "body": body,
            "bypassed_host_embedding": bypassed,
            "trainable": sum(p.numel() for p in self.trainable_parameters()),
            "total": body + emb + out,
        }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


@torch.no_grad()
def mean_of_pieces_init(
    model: TransplantLM, arm, host_tok, host_emb: torch.Tensor, verbose: bool = True
) -> dict:
    """Init each new stem row as the mean of the host's rows for its pieces.

    Rows are then rescaled to the host table's mean row norm: averaging shrinks
    vectors toward the centroid, and handing the frozen body inputs with the
    wrong scale is a gratuitous handicap.
    """
    V = model.heads["stem"]
    d = host_emb.shape[1]
    new = torch.zeros(V, d, dtype=torch.float32)
    misses = 0
    for vid in range(V):
        s = piece_string(arm, vid)
        ids = host_tok(s, add_special_tokens=False)["input_ids"] if s else []
        if ids:
            new[vid] = host_emb[torch.tensor(ids)].float().mean(0)
        else:
            misses += 1
            new[vid] = host_emb[torch.randint(0, host_emb.shape[0], (1,))].float()[0]

    target_norm = host_emb.float().norm(dim=1).mean()
    cur = new.norm(dim=1, keepdim=True).clamp(min=1e-6)
    new = new / cur * target_norm

    model.emb["stem"].weight.copy_(new.to(model.emb["stem"].weight.dtype))
    stats = {
        "init": "mean_of_pieces",
        "vocab": V,
        "empty_pieces": misses,
        "host_mean_row_norm": float(target_norm),
    }
    if verbose:
        print(f"  mean-of-pieces init: {stats}", flush=True)
    return stats


@torch.no_grad()
def random_init(
    model: TransplantLM, host_emb: torch.Tensor, verbose: bool = True
) -> dict:
    """Gaussian init matched to the host table's per-element std."""
    std = float(host_emb.float().std())
    model.emb["stem"].weight.normal_(0.0, std)
    if verbose:
        print(f"  random init: std={std:.5f}", flush=True)
    return {"init": "random", "std": std}


@torch.no_grad()
def native_init(model: TransplantLM, host_emb: torch.Tensor) -> dict:
    """Copy the host's own table — the untouched-pretrained reference arm."""
    model.emb["stem"].weight.copy_(host_emb.to(model.emb["stem"].weight.dtype))
    return {"init": "native"}


def load_host(model_id: str, dtype=torch.bfloat16, device="cuda"):
    """Return (body, embedding_table, tokenizer, hidden_size) for a host model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    body = lm.model  # embed_tokens + layers + norm; we bypass embed_tokens
    host_emb = lm.get_input_embeddings().weight.detach().clone()
    tok = AutoTokenizer.from_pretrained(model_id)
    body = body.to(device)
    body.eval()
    return body, host_emb, tok, lm.config.hidden_size
