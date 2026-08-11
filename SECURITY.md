# Security

## Reporting

Report suspected vulnerabilities via GitHub private vulnerability reporting on
this repository. Please do not open public issues for security reports.

## Operational scope — honest statement (Phase 1)

BM25 retrieval, a persisted SQLite index, and citation-bearing search work
end-to-end locally (Phase 1) against file/directory input, contained to an
allowed base directory via ported `path_safety` (`ensure_within_base`).
Embedding providers (Ollama, OpenAI-compatible) exist and are tested but are
not yet consumed by retrieval — nothing sends document content to a cloud
provider today. The REST API and MCP server, the network-facing surface a
remote attacker would need, are Phase-4 stubs. Today's attack surface is
local file ingestion plus the development toolchain and dependency supply
chain, mitigated by: the loader path containment above, pinned dev
dependencies with a lockfile-parity CI gate (`uv sync --locked`),
`pip-audit` in CI, and gitleaks in CI and pre-commit.

This section is rewritten as each phase lands real surface, per SPEC.md §7:

- Mutating REST routes sit behind a shared-secret header (constant-time
  comparison) and are disabled when the secret is unset; the service binds
  127.0.0.1 by default (Phase 4).
- SSRF guard on URL ingestion and cloud-provider endpoints, with the local
  Ollama endpoint as the one named exception (Phase 4). Not yet implemented:
  Phase 1 ships the cloud-provider surface (`OpenAICompatibleEmbedder`,
  operator-set `EmbeddingConfig.base_url`) but no validation rejecting
  loopback/private/link-local/IPv6-mapped endpoints and no explicit redirect
  policy. Exploitability today is low — no network-facing caller can set
  `base_url`; the REST/MCP surface that would expose one is a Phase-4 stub —
  but the guard itself does not exist yet.
- Rate limiting, when it arrives, is process-local — not a distributed or
  DoS-grade control, and it will be documented as such.
- Redaction at the LLM boundary (names → tokens, configurable patterns) runs
  before text leaves the process for a cloud provider; local mode sends
  nothing anywhere (Phase 5). Redaction is pattern-based and does not
  guarantee removal of all sensitive content.
