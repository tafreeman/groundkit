<!-- Transcluded from the repo-root SECURITY.md — GitHub reads that path for
     the repository's security policy, so it has to stay at the root, and a
     copy here would be a second source of truth for the disclosure contact.
     The included file supplies this page's H1. -->

--8<-- "SECURITY.md"

---

## Where text can leave the process

The security policy above covers disclosure and operational scope. For the
data-flow question — which paths transmit your document text, to where, and
what is redacted before they do — see
[The LLM boundary](architecture/llm-boundary.md). It is an exhaustive
inventory, and it is explicit about which boundary the redaction pass
covers: cloud **chat** egress, and only that one. The cloud **embedding**
boundary is deliberately unredacted — a recorded deviation, not an
unfinished feature (ADR-0017 decision 5) — so a cloud embedding provider
sees your document and query text in the clear regardless of redaction
configuration.
