# Eval harness

Most of the harness is reached through `grk eval` — the [eval harness
guide](../guides/evals.md) covers what it measures, what it deliberately does
not, and how to reproduce any report it writes.

The two modules below have no CLI verb and are called directly from Python.
That is deliberate in both cases: one *writes* a corpus, which the read-only
eval surface is not the place for, and the other answers a question about a
report after the fact rather than during a run.

## Significance

`significant` is decided by the percentile confidence interval and never by
`p_value_two_sided`; the two are computed independently. That p-value is a
Monte Carlo achieved significance level over the observed deltas rather than a
null-centred p-value, and it carries a `(r + 1) / (B + 1)` finite-sample
correction — so it has a floor and can never be reported as exactly zero.

The unit of resampling is a query. Pairing is checked by query id, so a
missing or reordered query raises rather than silently becoming an unpaired
comparison, and the generator is seeded per call so a result does not depend
on call order.

::: groundkit.evals.significance

## BEIR adaptation

BEIR relevance is per document and a groundkit judgment is a verbatim quote,
so each adapted gold quote is the whole document text — document-level
relevance expressed in the span vocabulary, rather than span annotations BEIR
does not provide. A score computed over the adapted set is therefore **not**
comparable to a published BEIR number, which ranks documents where `grk eval`
ranks chunks; the guide's [warning on that
point](../guides/evals.md#running-the-harness-over-another-corpus) states what
comparing them would require.

A BEIR corpus is an untrusted third-party download, and every identifier that
becomes part of a path is treated as such — validated as a single path
component, containment-checked independently, and rejected before anything is
written rather than after.

::: groundkit.evals.beir
