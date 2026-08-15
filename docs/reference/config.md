# Configuration

Typed configuration, frozen and fail-closed: unknown keys raise at
construction time and there is no lenient mode.

API keys are never configuration values. `EmbeddingConfig` carries the *name*
of the environment variable to read at call time, so a key cannot end up
serialized into a config dump, a log line, or an index manifest (SPEC.md §7).

!!! warning "`base_url` decides where your text goes — not `provider`"

    `EmbeddingConfig.base_url` is unconstrained operator input. Pointing the
    `ollama` provider at a remote host sends full document text over the
    network while the config still reads as local. See
    [The LLM boundary](../architecture/llm-boundary.md).

::: groundkit.config
