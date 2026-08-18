# Contributing to groundkit

groundkit is spec-driven: [`SPEC.md`](SPEC.md) is the contract, and deviating
from it requires an ADR proposal first (see [ADRs](#adrs) below). This file is
the other half — the conventions that are enforced by CI or by a test but are
not written down anywhere a fresh clone actually ships. Notably, they are
**not** in `CLAUDE.md`: that file is gitignored on purpose (a solo-dev-machine
convenience, not repo content — see the comment above its entry in
`.gitignore`), so a contributor who has never worked in this repo with an
agent attached would otherwise learn these rules only by reading `SPEC.md`
end to end or by breaking a CI gate. This file exists so neither is required.

`CLAUDE.md` is agent instruction, not contributor guidance, which is why it's
untracked rather than merely undocumented: a fresh clone will not have it at
all, and that absence is expected, not a missing file. The contract you're
bound by either way is [`SPEC.md`](SPEC.md), whose §9 is the phase authority
regardless of what any agent-instruction file claims.

## Setup

Python 3.11+, [`uv`](https://docs.astral.sh/uv/) for the environment and
lockfile.

```bash
uv sync --group dev
```

This installs the `dev` dependency group, which — unlike a typical Python
project — already includes `lancedb`, the OpenTelemetry SDK and both OTLP
exporters, `pypdf`, and `beautifulsoup4`. Those are optional *extras* for an
end user (`dense`, `otel`, `pdf`, `html`), pinned into `dev` anyway so that
running the test suite locally exercises the same backends CI does, rather
than skipping them because an extra happens to be absent. `rerank`
(`sentence-transformers`) is the deliberate exception — it pulls in torch, a
multi-gigabyte install — see [Gated workflows](#gated-workflows-not-part-of-a-normal-pr)
below.

**Gotcha, and it is written down nowhere else:** if you have a
`grk serve-mcp` server running against this checkout — e.g. a Claude
Desktop/Code MCP client connected to it — it holds a lock on
`.venv/Scripts/grk.exe`, and `uv sync` fails with `os error 32` (the file is
in use) trying to relink it. Stop the server first if you can. If you can't
— it may belong to someone else using this checkout, and killing it is not
your call — the fix depends on what you're trying to do:

- **Re-running a sync you've already done once** (nothing new to install):
  skip the sync and run against the environment as it stands.

  ```bash
  uv run --no-sync pytest ...
  ```

- **Installing a group/extra you don't have yet** (e.g. `docs` on a fresh
  clone, to build the site): `--no-sync` can't help, because there's nothing
  to run yet — and `uv sync ... --no-install-project` still fails the same
  way, because uv removes and relinks the project's own entry-point script
  as part of reconciling the environment regardless. Update the lockfile,
  then install only the *dependencies* via `uv pip`, which never touches the
  project's own installed script:

  ```bash
  uv lock
  uv export --no-emit-project --group docs --extra dense --format requirements-txt -o /tmp/reqs.txt
  uv pip install -r /tmp/reqs.txt
  ```

  This was exercised directly while writing this file: `uv sync --group docs
  --extra dense` reproduced `os error 32` against a live `grk serve-mcp`
  process, and the three commands above installed the same packages into the
  same `.venv` without touching `grk.exe` or disturbing the running server.

## Before you open a PR: the gates

`.github/workflows/ci.yml` has seven jobs, all required, none
`continue-on-error`. Run the parts that matter for your change locally before
pushing — `pre-commit` (installed by the `dev` group; run `pre-commit install`
once) covers `ruff check --fix`, `ruff format`, and `mypy` as a **local hook
through the project venv**, deliberately, so the file set pre-commit checks
matches what CI checks rather than a narrower set `mirrors-mypy` would see.
`pre-commit` also runs `gitleaks`.

```bash
uv sync --locked --group dev          # lint / typecheck / test jobs
uv run ruff check .
uv run ruff format --check .
uv run mypy                           # strict; files = src/groundkit + tests
uv run pytest --cov --cov-report=term
uv run coverage report                # whole-package 80% gate — see below
```

- **lint** — `ruff check .` then `ruff format --check .`. Line length 100,
  the full rule set is in `pyproject.toml`'s `[tool.ruff.lint]`. `tests/**`
  is exempted from `S101` (bare `assert` is how pytest works) and `ARG002`
  (a protocol-conformant test fake has to keep the protocol's exact parameter
  names, whether or not the fake uses them — see
  [Protocol seams](#protocol-seams-use-assert_signature_parity-not-isinstance)).
- **typecheck** — `mypy` in strict mode over `src/groundkit` and `tests`
  together (`mypy_path` includes `tests` so cross-test helper imports, like
  `assert_signature_parity`, resolve under mypy the way pytest resolves them
  at runtime — without that the checked file set silently diverges from the
  executed one).
- **test** — the full suite on Python 3.11, 3.12 and 3.13, plus **two**
  coverage gates. See [below](#two-coverage-gates-and-why-both-exist).
- **docs** — `uv sync --locked --group docs --extra dense && uv run mkdocs build --strict`.
  `--strict` promotes every MkDocs warning to a build failure: a relative
  link to a page that no longer exists, a `#fragment` no heading produces,
  and — the recurring case — a new page under `docs/` that nobody added to
  `nav` in `mkdocs.yml`. The `dense` extra isn't decoration here either:
  `mkdocstrings` imports groundkit to render the API reference, so a missing
  optional dependency would surface as a silently thin page instead of a
  build error. Build it locally the same way before touching anything under
  `docs/`:

  ```bash
  uv sync --group docs --extra dense
  uv run mkdocs build --strict
  ```

  If that first `uv sync` hits the `os error 32` MCP-server lock, that's the
  "installing a group you don't have yet" case in the [setup gotcha](#setup)
  above — use `uv lock` + `uv pip install` from the exported requirements,
  then `uv run --no-sync mkdocs build --strict`.

  Two things `--strict` does **not** protect you from, both learned the hard
  way on 2026-08-18:

  1. **It only governs Markdown pages.** MkDocs copies every *non-Markdown*
     file in `docs_dir` into the built site verbatim, and
     `validation.omitted_files` never sees them. A PDF dropped under `docs/`
     raises no warning, survives the strict build, and is published to the
     public site. Review output and working artifacts belong in the
     gitignored `_audit/` at the repo root, never under `docs/`.
  2. **The site's deploy path is repository state, not repo content.**
     GitHub Pages must have its source set to **"GitHub Actions"**
     (`build_type: workflow`), because `docs.yml` publishes with
     `actions/deploy-pages`. If it is set to a branch instead, the build job
     still passes and the deploy job fails with a bare
     `HttpError: Not Found`, *and* GitHub re-arms its own Jekyll builder to
     race the real deploy. Nothing in the tree can assert this, so if the
     `docs` workflow starts failing only in its `deploy` job, check
     `gh api repos/tafreeman/groundkit/pages --jq .build_type` first.
- **infra** — validates the Dockerfile builds and runs as uid 10001, that
  Kubernetes manifests render via `kubectl kustomize`, and that the Terraform
  module is formatted and `validate`s. Needs Docker, `kubectl` and Terraform
  locally to reproduce; most contributors will only touch this job's territory
  if they're changing `infra/`.
- **audit** — `pip-audit` against a lockfile-derived `requirements-audit.txt`.
- **secrets** — `gitleaks` over the PR's commits, allowlisted via
  `.gitleaks.toml` for known planted test sentinels only.

### Gated workflows (not part of a normal PR)

`eval-gated.yml` (`EVAL_GATED=1 uv run pytest tests/test_eval_gated.py -v`)
and `rerank-gated.yml` (`RERANK_GATED=1 uv run pytest tests/test_rerank_gated.py -v`,
plus `tests/test_eval_rerank_gated.py`) run real-model paths — a live Ollama
for the first, `sentence-transformers`/torch for the second — on a weekly
schedule or when a PR is labelled `eval-gated`/`rerank-gated`. They are never
`continue-on-error`, because each is the *sole* proof of its backend
(everything in `retrieval/rerank.py` that doesn't need a model is pure and
already covered by the default suite), but they also never block a normal
PR by default — pulling a model or a multi-gigabyte torch wheel on every push
would be the wrong cost to impose on everyone. Label your PR with one of
those names if your change touches the path it proves.

## Two coverage gates, and why both exist

CI's `test` job runs `coverage report` (whole-package, `fail_under = 80`,
`precision = 2` — 79.99% fails, it does not round up) **and** a second,
narrower gate, run exactly the way `ci.yml` runs it — the include pattern is
parsed out of `pyproject.toml` at run time rather than hand-copied, so this
command and the gate's current module list can never drift apart. Don't
paste today's module list into a script of your own; run this instead, so
it stays correct after the list changes:

```bash
PATTERN=$(uv run python -c "import tomllib; print(','.join(tomllib.load(open('pyproject.toml', 'rb'))['tool']['groundkit']['coverage']['core_subset']))")
uv run coverage report --include="$PATTERN" --fail-under=80
```

The reason a second gate exists at all: the
whole-package gate cannot see a well-covered *peripheral* module (an optional
provider like `providers/embeddings.py`) offsetting an under-tested *core*
one. A core module can sit at 60% covered and 90% overall coverage still
passes, silently, unless something is checking the core subset in isolation.

**Every module in `core_subset`, and every module left out of it, is argued
in the comment above the table in `pyproject.toml` — not just listed.** That
comment is itself a convention: if you add a module to `retrieval/`, it's
caught automatically by that package's glob. If you add a root-level module
that's structurally similar to `runtime.py` (decides whether an answer is
*correct* or *current*) or structurally similar to `telemetry.py` (its
absence changes nothing about what a response contains), the expectation is
that you extend that comment with the same kind of reasoning — a per-file
argument, not a blanket rule — rather than silently leaving the new module
out. `index/dense.py`'s entry is the model to follow: it states not just why
the module is in, but the *cost* of including it (a mixed file, two-thirds
shared filtering logic and one-third the LanceDB-backed store, so the gate
can't tell if one half is covering for the other) and why that cost was
accepted anyway.

## Every regression test must be shown to fail first

This is the single most distinctive rule in this repo (SPEC.md §8), and it
is not a formality. For any test that closes a **named defect** — an
ADR-0001 port hazard, a review finding, a bug fix — the procedure is:

1. Revert the *source* fix only (not the test): `git stash push -- <the files
   your fix touches>`.
2. Run the new test and watch it fail.
3. Restore the fix: `git stash pop`.
4. Run the test again and watch it pass.
5. Report both directions — "fails without the fix, passes with it" — not
   just the final green run.

Why this matters more here than in most repos: this codebase's recurring
defect classes — crash windows, cancellation paths, cross-run
reproducibility — are **unreachable on a non-failing path**. A test that
never actually exercises the broken branch will pass against both the buggy
code and the fixed code, and neither line coverage nor a green suite can
tell those two tests apart from the outside. The 2026-08-14 cancellation
rollback is the concrete case: a fix that itself introduced a
`ProgrammingError` on a closed connection (masking the original exception)
only surfaced because the new regression test was run against the
*pre-fix* code and produced a different failure than expected. A test that
had only ever been run against the post-fix code would have looked exactly
as green as a correct one.

## ADRs

Convention: `docs/adr/ADR-NNNN-<slug>.md`, four-digit zero-padded, one
decision per record, alternatives considered and recorded — not just the
decision taken. Every ADR is listed in [`docs/adr/index.md`](docs/adr/index.md).
**Deviating from `SPEC.md` requires an ADR proposal first**; this is not
optional process theatre — several ADRs in this repo exist specifically
because a port-time hazard or a review finding could not be closed inside
the existing spec text without contradicting it.

Note the filename shape is specific to this repo: sibling repos in this
portfolio use a bare `NNN-` or `NNNN-` prefix, or a different directory
entirely. Don't copy another repo's ADR convention here, and don't invent a
new one — follow the existing files under `docs/adr/`.

## Protocol seams use `assert_signature_parity`, not `isinstance`

Component boundaries in this repo are `typing.Protocol` classes
(`ingestion/protocols.py`, `index/protocols.py`, `retrieval/protocols.py`,
`providers/protocols.py`). `tests/test_protocol_conformance.py` exists
because `isinstance(impl, SomeProtocol)` against a `@runtime_checkable`
Protocol only checks that members of the same **name** exist — not their
parameter names, order, defaults, sync-vs-async, or types. A rename from
`query` to `_query` on an implementation (ADR-0001 hazard 4 — the actual
defect this closes, ported from ARP) would pass every `isinstance` check
untouched while breaking every real caller.

`assert_signature_parity(protocol, implementation)` checks, for every public
member the Protocol declares in its own class body: method-vs-property kind,
sync-vs-async, parameter names/order/kinds/defaults, and resolved type hints
via `typing.get_type_hints`. If you add a new Protocol seam, wire it through
this helper, not a bare `isinstance` assertion — a passing `isinstance`
check on a Protocol proves less than it looks like it proves.

## No number in any doc that wasn't generated

Don't hand-write a recall, nDCG, latency or coverage figure into a Markdown
file. SPEC.md §2 permits a number in a doc only when it comes from a
generated eval artifact (`evals/results/*.json`, gitignored, regenerated by
`grk eval`) or a live badge — never a value typed in by hand, because a
hand-copied metric is a number that was true once and is unverifiable
thereafter. Corpus-size floors are the same rule in miniature: they're
asserted as literal numbers in `tests/test_corpus_integrity.py`, and that
test is the authoritative source, not `evals/README.md`'s prose description
of them.

## Commits & PRs

Conventional commits, derived from this repo's actual history — don't
invent a scope or type that doesn't appear below:

```
<type>(<scope>): <subject>
```

Types seen in this repo's log: `feat`, `fix`, `test`, `docs`, `ci`, `perf`,
`chore`. Subject in the imperative mood, no trailing period. Scope is
usually a package or concern (`ingestion`, `evals`, `index`, `providers`,
`terraform`, `secrets`, `limitations`), not a file name. Examples pulled
directly from `git log`:

```
fix(ingestion): refuse a credential in a URL query string
feat(ingestion): fetch http(s) URLs into verifiable local snapshots
perf(evals): reuse the scoring pass's retrieval for synthesis
fix(index): refuse a foreign SQLite file instead of writing schema into it
ci(secrets): allowlist the planted test sentinels via .gitleaks.toml
```

SPEC.md §8 asks for **small changesets** — each phase (and by extension, each
PR within one) ends with CI green and any docs it touches updated in the
same change, not a follow-up. Branch off `main`, open a PR back to it; this
repo does not currently maintain a long-lived integration branch separate
from `main` for day-to-day feature work.

### What `main` enforces

Since 2026-08-18 a repository ruleset makes this mechanical rather than a
convention, because until then every gate below was advisory — nothing stopped
a merge on red or a direct push:

- **No direct pushes, no force-pushes, no deletion.** Changes reach `main`
  through a pull request.
- **Nine required status checks**, all from `ci.yml`: `lint`, `typecheck`,
  `test (3.11)`, `test (3.12)`, `test (3.13)`, `docs`, `infra`, `audit`,
  `secrets`. The gated suites (`gated-eval`, `gated-rerank`) are deliberately
  **not** required — they skip by design without their env gate, and requiring
  a check that normally skips is how a branch becomes unmergeable.
- **Zero required approvals.** Deliberate, not an oversight: GitHub does not
  let you approve your own pull request, so on a single-maintainer repo any
  higher number locks the only maintainer out of their own `main`. Raise it
  the moment there is a second reviewer.
- **No bypass actors**, including administrators.

Two consequences worth knowing. Adding, renaming or removing a job in `ci.yml`
means updating the required-checks list in the same change, or `main` starts
requiring a check that no longer reports and nothing can merge. And the ruleset
targets `~DEFAULT_BRANCH` only — any other long-lived branch is unprotected.

## Where things live

- `src/groundkit/` — the package (src-layout; `pyproject.toml`'s
  `[tool.hatch.build.targets.wheel]` packages `src/groundkit`). Subpackages:
  `ingestion/`, `index/`, `retrieval/`, `providers/`, `service/`, `evals/`,
  plus root-level modules (`contracts.py`, `config.py`, `errors.py`,
  `runtime.py`, `identity.py`, `indexer.py`, `answer.py`, `extraction.py`,
  `snapshots.py`, `telemetry.py`, `cli.py`). `docs/reference/*.md` is the
  mkdocstrings-rendered API surface over the modules a developer is most
  likely to need from outside their own package — see those pages for the
  per-module reasoning rather than duplicating it here.
- `tests/` — flat, `tests/*.py`, no subdirectories. If you ever see a
  `tests/unit/`, `tests/graders/` or similar subdirectory in this repo, it is
  not groundkit structure.
- `evals/` — the golden corpus and its authoring contract
  (`evals/README.md` — **never** `evals/corpus/README.md`, which would
  silently become an indexed, retrievable corpus document), `judgments.jsonl`,
  and gitignored `results/`. This is the harness SPEC.md §8 calls "before
  features" — it landed ahead of hybrid retrieval and rerank specifically so
  every later feature reports a measured delta against it.
- `infra/` — Dockerfile, compose stack, Kubernetes manifests, Terraform
  module (Phase 6). Not scaffolding: each path is exercised by CI's `infra`
  job or has been verified by hand, with the verification recorded in
  `infra/README.md`.
- `docs/` — the MkDocs site. `docs/adr/` holds the ADRs; `docs/specs/`
  holds feature specs with open work still described in them (as opposed to
  `SPEC.md`, the single living contract for what's *done*).
