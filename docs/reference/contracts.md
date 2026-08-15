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

::: groundkit.contracts
