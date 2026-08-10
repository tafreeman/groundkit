# Known limitations

Honest and current, per repo policy. Updated with each phase.

## Current state (Phase 0)

Nothing is implemented yet. The repo contains the spec, ADR-0001, and a typed
skeleton whose modules are docstring-only placeholders. The `grk` CLI installs
and reports its version; it does nothing else.

## Deliberately out of scope for v1 (will not be built)

- Multi-tenant auth — single-operator service; the shared-secret header on
  mutating routes is not a tenancy model.
- Distributed indexing — one node, file-based index.
- Fine-tuning of embedding or rerank models.
- Agent loops — this is a retrieval service, not an agent runtime.
- UI beyond the docs site.
- GraphRAG.
- Broad vector-DB support — LanceDB (dense) + SQLite (metadata) behind
  interfaces; pgvector is a designed-for extension point, not a v1 feature.
