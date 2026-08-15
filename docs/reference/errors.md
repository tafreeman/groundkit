# Errors

Every failure groundkit raises is one of these types, and every one of them
descends from `GroundkitError` — so a caller can catch the whole surface
without catching `Exception`.

The hierarchy is the *fail closed* principle made concrete. An unconfigured
provider raises `ProviderNotConfiguredError` rather than falling back to
another one; a collection whose embedding identity does not match the
retriever's raises `IndexIdentityError` rather than searching anyway. There is
no cross-provider embedding fallback in groundkit at all: mixed semantic
spaces corrupt an index silently, and a typed error is the only honest
response (SPEC.md §2).

::: groundkit.errors
