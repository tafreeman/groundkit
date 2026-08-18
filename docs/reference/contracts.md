# Contracts

The data model. Every model here is a frozen Pydantic v2 model with
`extra="forbid"` — an unknown key is a construction-time error, not a field
that gets quietly dropped (SPEC.md §2, *fail closed*).

`Chunk` carries character offsets into its source document, and `Citation`
carries document ID + chunk ID + offsets. Together they are what makes
"citations are verifiable" a checkable property rather than a claim:
[`verify_citation`](retrieval.md#groundkit.retrieval.citations.verify_citation)
re-reads the source file and confirms the span still says what the citation
says it says.

`RetrievalResult` carries `source_class` and `extractor` (ADR-0016), which
route citation verification to the right re-derivation path. Their defaults
— `"text"` and `None` — exist for backward compatibility with a published
contract, not as a value any construction site should take implicitly: an
AST-level test asserts that every `RetrievalResult(...)` in `src/groundkit/`
passes both fields explicitly. `evals/` is exempt, and deliberately — the two
sites there build synthetic fixtures whose provenance is invented for the
harness rather than carried through from a store record, so there is nothing
upstream of them for a default to silently replace.

::: groundkit.contracts
