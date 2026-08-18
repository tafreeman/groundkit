# Extraction & snapshots

Two leaf modules shared by callers on opposite ends of a document's life
(ADR-0016): the same extractor that built a document's content at ingest
time is what
[`retrieval.citations.resolve_citation`](retrieval.md#groundkit.retrieval.citations.resolve_citation)
calls again to re-derive that text when verifying a citation later. One
implementation per format, used from both sides, is what makes "the same
extractor that produced it" (ADR-0016 decision 2) hold *by construction*
rather than by two independent implementations kept in sync by hand. Neither
module imports the other, and neither is imported back — both follow
`identity.py`'s placement precedent: a module outside whichever caller
happened to need it first, so sharing it creates no dependency edge between
`ingestion/` and `retrieval/`.

## Extraction

`PdfExtractor` and `HtmlExtractor` are imported by exactly two kinds of
caller: the PDF/HTML loaders, to build `Document.content` at ingest time,
and the citation resolver above. Neither `pypdf` nor `beautifulsoup4` is
imported at module level — importing `groundkit.extraction` never requires
either optional extra; only constructing the corresponding extractor does.

::: groundkit.extraction

## Snapshots

Pure path arithmetic, no I/O: where a URL-ingested document's local snapshot
lives on disk (ADR-0016 decision 4; see also
[ADR-0023](../adr/ADR-0023-snapshot-lifecycle-is-bound-to-the-document-row.md)
for when a snapshot is removed, not just where it lives). Written by the URL
loader and read by the citation resolver above, so the naming convention
that ties a document ID to its snapshot path is asserted from exactly one
place, and the two sides cannot independently drift on what it means.

::: groundkit.snapshots
