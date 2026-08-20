"""Structural guard for SPEC.md §2's central claim: "deterministic core, LLM
at the boundary".

The property currently holds, but only by review. A strictly *narrower*
version already has a mechanism: ``tests/test_service_tools.py``'s
``test_service_package_imports_no_write_path`` AST-scans the service
package for the same class of boundary violation. Nothing scans the
retrieval path itself, even though it is the property SPEC.md §2 actually
names. ``answer.py`` composes retrieval with synthesis, rewrite and the
judge one import away, so the obvious next feature request -- make query
rewrite an option on ``Retriever.search`` so ``grk search`` benefits too --
is a two-line change that would violate the project's central claim, and
nothing today would catch it.

Guard, demonstrated by injection (matching ``test_service_tools.py``'s own
convention): every file this scans was already correct when this test was
written, so there is no unfixed version to revert to per the SPEC.md §8
procedure. Add a barred import to any scanned file, watch this fail, remove
it.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "groundkit"

#: Every file the deterministic core comprises: three whole directories
#: (retrieval, index, ingestion) plus three top-level modules that sit
#: beneath them in the dependency graph but must equally never reach up
#: into the LLM boundary.
_DETERMINISTIC_CORE_FILES: tuple[Path, ...] = (
    *sorted((_SRC_ROOT / "retrieval").glob("*.py")),
    *sorted((_SRC_ROOT / "index").glob("*.py")),
    *sorted((_SRC_ROOT / "ingestion").glob("*.py")),
    _SRC_ROOT / "indexer.py",
    _SRC_ROOT / "contracts.py",
    _SRC_ROOT / "identity.py",
)

#: Exact module names barred outright.
_BARRED_EXACT: frozenset[str] = frozenset(
    {
        "groundkit.providers.llm",
        "groundkit.providers.synthesis",
        "groundkit.providers.query_rewrite",
    }
)

#: Barred by prefix: the whole eval harness, not just its judge -- a future
#: module added under groundkit.evals is barred by construction, not by
#: someone remembering to add it to _BARRED_EXACT.
#:
#: groundkit.providers.protocols is deliberately NOT barred, by omission:
#: the embedding seam legitimately sits in the retrieval path (Retriever
#: composes against EmbeddingProtocol), and that seam is the boundary
#: interface itself, not an LLM call.
_BARRED_PREFIX: str = "groundkit.evals"


def _is_barred(module_name: str) -> bool:
    return (
        module_name in _BARRED_EXACT
        or module_name == _BARRED_PREFIX
        or module_name.startswith(_BARRED_PREFIX + ".")
    )


def test_deterministic_core_imports_no_llm() -> None:
    """AST scan: no module in the deterministic core reaches the LLM boundary.

    The file list is checked for sanity before it is trusted: a
    ``_DETERMINISTIC_CORE_FILES`` silently resolving to nothing (a moved
    directory, a typo'd path) would make the scan below pass vacuously --
    the same failure mode ``assert_signature_parity`` guards against for a
    Protocol with no checkable members.

    Guard, demonstrated by injection: add ``from groundkit.providers.llm
    import build_chat`` (or any other barred import) to any file this scans
    and the assertion below fails, naming the offending file and module.
    """
    assert len(_DETERMINISTIC_CORE_FILES) >= 15, (
        "the file list looks too small to be the real deterministic core -- "
        "check _SRC_ROOT and the three glob patterns"
    )
    for known in (
        _SRC_ROOT / "retrieval" / "search.py",
        _SRC_ROOT / "index" / "bm25.py",
        _SRC_ROOT / "ingestion" / "chunking.py",
        _SRC_ROOT / "indexer.py",
        _SRC_ROOT / "contracts.py",
        _SRC_ROOT / "identity.py",
    ):
        assert known in _DETERMINISTIC_CORE_FILES, f"{known} missing from the scanned file list"

    offenders: list[str] = []
    for path in _DETERMINISTIC_CORE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module is not None and _is_barred(node.module):
                    offenders.append(f"{path.relative_to(_SRC_ROOT)} -> {node.module}")
            elif isinstance(node, ast.Import):
                offenders.extend(
                    f"{path.relative_to(_SRC_ROOT)} -> {alias.name}"
                    for alias in node.names
                    if _is_barred(alias.name)
                )
    assert not offenders, f"deterministic core reaches the LLM boundary: {offenders}"
