"""Typed exception hierarchy. Unconfigured provider or malformed output is a
typed error, never a silent fallback or coercion.

Phase 1 fills this per SPEC.md §2 (fail closed) and the ADR-0001 errors.py row.
"""
