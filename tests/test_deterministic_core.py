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

## The second guard: the eval harness is not a runtime dependency

``test_runtime_surface_does_not_import_the_eval_harness`` scans a wider set
for a narrower thing -- *every* runtime module, for any import of
``groundkit.evals``. The two are separate properties and are kept separate
deliberately. The first is about what the deterministic core may reach
(SPEC.md §2); the second is about layering, and it applies to the LLM
boundary too: ``providers/synthesis.py`` is allowed to call a model and is
still not allowed to import the harness.

Unlike the first, the second has an unfixed version to revert to.
``answer.py`` imported ``groundkit.evals.judge`` until GK-021 moved
``FaithfulnessJudge`` to ``providers/judge.py``, which is why the judge is
in ``_BARRED_EXACT`` below rather than covered by the eval-package prefix.
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

#: The eval harness package. Named once and used twice: barred by prefix
#: from the deterministic core below (the whole harness, not just its judge
#: -- a future module added under groundkit.evals is barred by construction,
#: not by someone remembering to add it to _BARRED_EXACT), and barred from
#: the whole runtime surface by the second test in this module.
_EVAL_PACKAGE: str = "groundkit.evals"

#: Exact module names barred outright.
#:
#: groundkit.providers.judge joined this set when GK-021 moved
#: FaithfulnessJudge out of groundkit.evals: the prefix rule above used to
#: cover the judge for free, and stopped covering it the moment it became a
#: provider. Its being a provider is precisely why it must stay barred --
#: the core may not call a model, wherever the module that calls one lives.
#:
#: groundkit.providers.protocols is deliberately NOT barred, by omission:
#: the embedding seam legitimately sits in the retrieval path (Retriever
#: composes against EmbeddingProtocol), and that seam is the boundary
#: interface itself, not an LLM call.
_BARRED_EXACT: frozenset[str] = frozenset(
    {
        "groundkit.providers.judge",
        "groundkit.providers.llm",
        "groundkit.providers.synthesis",
        "groundkit.providers.query_rewrite",
    }
)

#: The one runtime module exempt from the eval-harness scan, and the only one
#: that may be: cli.py hosts ``grk eval``, so the harness entry point is
#: supposed to reach the harness. A single named file rather than a list, so
#: widening the exemption is a visible edit here.
_HARNESS_ENTRY_POINT: Path = _SRC_ROOT / "cli.py"

#: Every runtime module: all of src/groundkit except the eval package itself
#: and the harness entry point above.
_RUNTIME_SURFACE_FILES: tuple[Path, ...] = tuple(
    sorted(
        path
        for path in _SRC_ROOT.rglob("*.py")
        if "evals" not in path.relative_to(_SRC_ROOT).parts and path != _HARNESS_ENTRY_POINT
    )
)

#: Floor for the vacuity check on _RUNTIME_SURFACE_FILES -- comfortably below
#: the real count, high enough that a glob resolving to one subtree, or to
#: nothing, fails loudly instead of passing over an empty scan.
_MIN_RUNTIME_SURFACE_FILES: int = 30


def _is_eval_package(module_name: str) -> bool:
    """Whether ``module_name`` is the eval harness package or a module inside it."""
    return module_name == _EVAL_PACKAGE or module_name.startswith(_EVAL_PACKAGE + ".")


def _is_barred(module_name: str) -> bool:
    return module_name in _BARRED_EXACT or _is_eval_package(module_name)


def _imported_modules(path: Path) -> list[str]:
    """Every absolute module name ``path`` imports, across both import forms.

    Args:
        path: The source file to parse.

    Returns:
        Module names exactly as written: ``from x.y import z`` contributes
        ``x.y``, ``import x.y`` contributes ``x.y``. Relative imports
        contribute nothing — this package uses absolute imports throughout,
        so a relative one cannot name another top-level package and needs no
        resolution here.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module is not None:
                modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


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

    offenders = [
        f"{path.relative_to(_SRC_ROOT)} -> {module}"
        for path in _DETERMINISTIC_CORE_FILES
        for module in _imported_modules(path)
        if _is_barred(module)
    ]
    assert not offenders, f"deterministic core reaches the LLM boundary: {offenders}"


def test_runtime_surface_does_not_import_the_eval_harness() -> None:
    """AST scan: no runtime module imports ``groundkit.evals`` (GK-021).

    This is a layering guard, not a packaging assertion: ``groundkit.evals``
    ships inside the wheel today. What it protects is the *option* — the
    harness can become an extra, or be dropped from a runtime install, only
    for as long as no module on the library import path reaches into it. One
    production import is enough to take that option away, and an import is a
    one-line change that reviews as convenience.

    ``cli.py`` is exempt and is the only file that may be: ``grk eval`` is
    the harness entry point, so the CLI reaching the harness is the design,
    not a leak. Everything else is scanned, the LLM boundary included —
    ``providers/synthesis.py`` may call a model and still may not import the
    harness.

    Guard, demonstrated by revert (SPEC.md §8): ``git stash push
    src/groundkit/answer.py`` restores the pre-GK-021 import of
    ``groundkit.evals.judge`` and this fails with ``answer.py ->
    groundkit.evals.judge`` in the offender list; ``git stash pop`` restores
    the pass. The two sanity assertions below hold in both directions, so
    the failure is the one this test is for.
    """
    assert len(_RUNTIME_SURFACE_FILES) >= _MIN_RUNTIME_SURFACE_FILES, (
        "the file list looks too small to be the whole runtime surface -- "
        "check _SRC_ROOT and the rglob in _RUNTIME_SURFACE_FILES"
    )
    assert _HARNESS_ENTRY_POINT.is_file(), (
        f"{_HARNESS_ENTRY_POINT} does not exist, so the one exemption is "
        "excluding nothing and this scan's only blind spot is unexplained"
    )
    for known in (
        _SRC_ROOT / "answer.py",
        _SRC_ROOT / "providers" / "synthesis.py",
        _SRC_ROOT / "retrieval" / "search.py",
        _SRC_ROOT / "service" / "tools.py",
    ):
        assert known in _RUNTIME_SURFACE_FILES, f"{known} missing from the scanned file list"

    offenders = [
        f"{path.relative_to(_SRC_ROOT)} -> {module}"
        for path in _RUNTIME_SURFACE_FILES
        for module in _imported_modules(path)
        if _is_eval_package(module)
    ]
    assert not offenders, (
        "runtime code imports the eval harness, which makes groundkit.evals "
        f"undroppable from a library install (GK-021): {offenders}"
    )
