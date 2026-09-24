"""Qwen3-1.7B with LoRA and, in the tokens arm, an invariant-feature encoder.

The language model is frozen; LoRA adapters on the attention and MLP
projections are trained in both arms with the same configuration. In the
`tokens` arm the tokenizer gains one special token `<st>` and every occurrence
of it in the input has its embedding replaced by the encoder's output for the
corresponding object (in sequence order). The `table` arm has no encoder and
no `<st>` — the state is plain text.

Adapters (LoRA A/B weights plus the encoder) are saved as one `.pt` file; the
base weights are never written.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .configs import ARM_TOKENS, V1Config
from .encoder import InvariantEncoder, apply_lora, lora_state_dict
from .sequence import ST


def load_tokenizer(cfg: V1Config):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True)
    tok.add_special_tokens({"additional_special_tokens": [ST]})
    return tok


class V1Model(nn.Module):
    def __init__(
        self,
        cfg: V1Config,
        arm: str,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lm: nn.Module | None = None,
        tokenizer=None,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.cfg = cfg
        self.arm = arm
        self.tok = tokenizer if tokenizer is not None else load_tokenizer(cfg)
        self.st_id = int(self.tok.convert_tokens_to_ids(ST))
        self.pad_id = int(self.tok.pad_token_id)
        self.lm = (
            lm
            if lm is not None
            else AutoModelForCausalLM.from_pretrained(
                cfg.base_model, torch_dtype=dtype, local_files_only=True
            )
        )
        if self.lm.get_input_embeddings().num_embeddings <= self.st_id:
            self.lm.resize_token_embeddings(len(self.tok))
        self.lm.to(device)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        if cfg.gradient_checkpointing and hasattr(
            self.lm, "gradient_checkpointing_enable"
        ):
            self.lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # checkpointing needs a grad-requiring input; the LoRA path provides
            # one only after the first layer, so ask the embeddings to pass grads
            if hasattr(self.lm, "enable_input_require_grads"):
                self.lm.enable_input_require_grads()
        self.lora_params = apply_lora(
            self.lm, cfg.lora_targets, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout
        )
        d_model = self.lm.get_input_embeddings().embedding_dim
        self.encoder = (
            InvariantEncoder(cfg.d_enc, d_model, cfg.enc_layers).to(device)
            if arm == ARM_TOKENS
            else None
        )
        self.device = torch.device(device)

    # -- parameters ---------------------------------------------------------

    def param_groups(self) -> list[dict]:
        groups = [{"params": self.lora_params, "lr": self.cfg.lr_lora}]
        if self.encoder is not None:
            groups.append(
                {"params": list(self.encoder.parameters()), "lr": self.cfg.lr_encoder}
            )
        return groups

    def n_trainable(self) -> int:
        n = sum(p.numel() for p in self.lora_params)
        if self.encoder is not None:
            n += sum(p.numel() for p in self.encoder.parameters())
        return n

    # -- embeddings ---------------------------------------------------------

    def state_embeddings(
        self,
        self_feats: torch.Tensor,
        pair_feats: torch.Tensor,
        pair_mask: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        assert self.encoder is not None, "the table arm has no state tokens"
        out = self.encoder(
            self_feats.to(self.device),
            pair_feats.to(self.device),
            pair_mask.to(self.device),
            pair_index.to(self.device),
        )
        return out.to(self.lm.get_input_embeddings().weight.dtype)

    def embed(
        self, input_ids: torch.Tensor, st_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Token embeddings with every `<st>` replaced, in row-major order."""
        input_ids = input_ids.to(self.device)
        embeds = self.lm.get_input_embeddings()(input_ids)
        positions = (input_ids == self.st_id).nonzero(as_tuple=False)
        if positions.shape[0] == 0:
            return embeds
        if st_embeds is None or st_embeds.shape[0] != positions.shape[0]:
            got = 0 if st_embeds is None else st_embeds.shape[0]
            raise ValueError(
                f"{positions.shape[0]} state tokens in the input but {got} embeddings"
            )
        embeds = embeds.clone()
        embeds[positions[:, 0], positions[:, 1]] = st_embeds.to(embeds.dtype)
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        self_feats: torch.Tensor | None = None,
        pair_feats: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        pair_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        st_embeds = None
        if self.encoder is not None and self_feats is not None and self_feats.shape[0]:
            st_embeds = self.state_embeddings(
                self_feats, pair_feats, pair_mask, pair_index
            )
        embeds = self.embed(input_ids, st_embeds)
        out = self.lm(
            inputs_embeds=embeds,
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device),
        )
        return out.loss

    # -- persistence --------------------------------------------------------

    def save_adapters(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"arm": self.arm, "lora": lora_state_dict(self.lm)}
        if self.encoder is not None:
            payload["encoder"] = {
                k: v.detach().cpu() for k, v in self.encoder.state_dict().items()
            }
        torch.save(payload, path)
        return path

    def load_adapters(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        if payload["arm"] != self.arm:
            raise ValueError(
                f"adapters are for arm {payload['arm']!r}, model is {self.arm!r}"
            )
        expected = set(lora_state_dict(self.lm))
        got = set(payload["lora"])
        if expected != got:
            missing = sorted(expected - got)[:3]
            extra = sorted(got - expected)[:3]
            raise ValueError(f"adapter keys differ: missing {missing}, extra {extra}")
        self.lm.load_state_dict(payload["lora"], strict=False)
        if self.encoder is not None:
            self.encoder.load_state_dict(payload["encoder"])
