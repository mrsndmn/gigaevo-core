"""The package boundary of ADR 0004: what `experiments/geostate` may import."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[2] / "experiments" / "geostate"

# The only modules of noregeo_synth that geostate may use; anything else — the
# guard/subprocess machinery in particular — must be copied, not imported.
NOREGEO_ALLOWLIST = {"credentials", "dataset", "configs", "figures", "verify"}

# The MLS launchers alone may reuse the shared job plumbing (path rewriting,
# staging, image and env constants) of the vocab-factorization launchers.
LAUNCHERS = {"run_e0_jobs.py", "run_v1_jobs.py"}
LAUNCHER_ALLOWLIST = {
    "experiments.vocab_factorization.jobs",
    "experiments.vocab_factorization.run_train_jobs",
}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # relative import inside the package
                names.append("." + module)
            else:
                names.extend(
                    f"{module}.{alias.name}" if module else alias.name
                    for alias in node.names
                )
    return names


def test_no_import_from_gigaevo_and_only_allowlisted_noregeo_modules():
    for path in sorted(PACKAGE.glob("*.py")):
        for name in _imports(path):
            assert not name.startswith("gigaevo"), f"{path.name} imports {name}"
            if name.startswith("experiments.noregeo_synth"):
                leaf = name.split(".")[2] if name.count(".") >= 2 else ""
                assert leaf in NOREGEO_ALLOWLIST, f"{path.name} imports {name}"
            if name.startswith("experiments.vocab_factorization"):
                module = ".".join(name.split(".")[:3])
                assert path.name in LAUNCHERS and module in LAUNCHER_ALLOWLIST, (
                    f"{path.name} imports {name}"
                )


def test_configs_is_standard_library_only():
    stdlib = set(sys.stdlib_module_names)
    for name in _imports(PACKAGE / "configs.py"):
        top = name.split(".")[0]
        assert top in stdlib, f"configs.py imports {name}"
