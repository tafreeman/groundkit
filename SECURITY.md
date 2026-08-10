# Security

## Reporting

Report suspected vulnerabilities via GitHub private vulnerability reporting on
this repository. Please do not open public issues for security reports.

## Operational scope — honest statement (Phase 0)

Nothing is implemented yet: the package contains typed, docstring-only
placeholders and a version-reporting CLI. The current attack surface is the
development toolchain and dependency supply chain only, mitigated by: pinned
dev dependencies with a lockfile-parity CI gate (`uv sync --locked`),
`pip-audit` in CI, and gitleaks in CI and pre-commit.

This section is rewritten as each phase lands real surface, per SPEC.md §7:

- Mutating REST routes sit behind a shared-secret header (constant-time
  comparison) and are disabled when the secret is unset; the service binds
  127.0.0.1 by default (Phase 4).
- SSRF guard on URL ingestion and cloud-provider endpoints, with the local
  Ollama endpoint as the one named exception (Phases 1/4).
- Rate limiting, when it arrives, is process-local — not a distributed or
  DoS-grade control, and it will be documented as such.
- Redaction at the LLM boundary (names → tokens, configurable patterns) runs
  before text leaves the process for a cloud provider; local mode sends
  nothing anywhere (Phase 5). Redaction is pattern-based and does not
  guarantee removal of all sensitive content.
