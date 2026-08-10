"""FastAPI service mirroring the MCP tools (Phase 4). Binds 127.0.0.1 by
default; mutating routes require the shared-secret header (constant-time
comparison) and are disabled when the secret is unset.
"""
