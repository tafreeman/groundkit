# Deployment

groundkit's deployment paths live in `infra/` — a container image, a compose
stack, Kubernetes manifests, and a Terraform module. This page is the map and
the warnings; `infra/README.md` in the repository is the operating manual, and
the reasoning behind each shape is in the ADRs linked below.

None of this is required to use groundkit. `grk ingest`, `grk search`,
`grk serve` and `grk serve-mcp` run on a laptop with no container runtime and no
credentials, which is what SPEC.md §10 makes the definition of done.

## Read this first: there is no authentication

Every operation the service exposes is read-only, which satisfies SPEC.md §7's
shared-secret requirement vacuously — the set of mutating operations is empty,
so the set requiring a secret is empty. The consequence is stated in
[ADR-0014](../adr/ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md)
decision 7 and it governs everything on this page: **the address the service is
reachable at is its entire access control.** A `search` response carries
document text and absolute source paths; `index_status` and `list_collections`
carry collection topology.

Locally that is enforced in the process — `grk serve` refuses a non-loopback
`--host` unless you acknowledge it. **In a container it cannot be**, because a
process bound to `127.0.0.1` inside a container is reachable from nothing: not
the host, not a sibling container, not a published port. So the guarantee moves
one layer out, and each deployment surface re-establishes it in its own terms
([ADR-0021](../adr/ADR-0021-container-exposure-and-filesystem-hardening.md)
decision 1):

| Surface | What keeps it private | How you reach it |
|---|---|---|
| compose | host-loopback publish (`127.0.0.1:8765:8765`) | `curl http://127.0.0.1:8765/…` |
| Kubernetes | `ClusterIP` Service — not NodePort, not LoadBalancer | `kubectl port-forward` |
| Terraform (AWS) | security group with no ingress rules at all | SSM port forwarding |

The image itself cannot enforce any of this. `docker run -p 8765:8765
groundkit` — the obvious shorter command — publishes an unauthenticated,
content-bearing surface on every interface of the host, and so does
`--network host`.

## The container image

Multi-stage, built from the repository root, running as a fixed non-root uid
(`10001`) with a read-only root filesystem. The uid is part of the image's
public contract rather than an implementation detail: a Kubernetes `runAsUser`,
a Docker named volume's inherited ownership, and a host bind mount all have to
name the same identity.

Three mounts, three different answers, and the differences are the point:

| Path | Mode | Why |
|---|---|---|
| `/` | read-only | Nothing the service does writes into the image. |
| `/data/corpus` | **read-only** | Citation resolution re-reads source files and never writes one. |
| `/data/index` | **read-write** | WAL sidecars, `CREATE TABLE IF NOT EXISTS`, and the open-time chmod. |

That third row is the one worth understanding.
[KNOWN_LIMITATIONS](../limitations.md) has long recorded that "read-only does
not mean the process writes no bytes," and deferred filesystem-level enforcement
to this phase. What that deferral actually buys, now that it is cashed, is the
**corpus** mount: the documents every citation resolves against are unwritable
by the serving process, enforced by the kernel rather than scoped in Python. The
index directory stays writable and no mount flag changes that while the store is
in WAL mode ([ADR-0002](../adr/ADR-0002-index-persistence.md)).

The image carries the `dense` extra and not `rerank`: torch is multiple
gigabytes for a capability `Retriever.search` cannot reach at all
([ADR-0012](../adr/ADR-0012-rerank-eval-stage-reorders-upstream-stage.md)
decision 2).

## The compose stack

Four services — groundkit, Ollama, an OpenTelemetry collector, and Jaeger — with
zero credentials and no host port bound anywhere but loopback. Ollama and the
collector publish nothing at all; they are reachable only from the stack's own
network.

One detail is easy to misread as a hole in the SSRF guard. The service reaches
Ollama at `http://ollama:11434`, which resolves to an RFC1918 bridge address,
not to loopback. That works because `OllamaEmbedder` sets
`_allow_private_endpoint` as a **class attribute** scoped to that one provider
(ADR-0014 decision 10) — not because the guard is relaxed. An OpenAI-compatible
endpoint pointed at a private address is still refused.

### Traces: wired, not yet emitting

Phase 6 lands in two changes. The first is purely additive infrastructure and
changes nothing under `src/`; the second adds the OpenTelemetry instrumentation
and the JSON log formatter. **Between the two, the collector and Jaeger are real
and correctly wired, and groundkit emits no spans** — so the Jaeger UI is empty
and that is expected rather than a misconfiguration. The collector→Jaeger leg is
provable on its own with a synthetic OTLP payload in the meantime.

When instrumentation does land, what a span may carry is an allowlist rather
than a denylist
([ADR-0022](../adr/ADR-0022-observability-dependency-shape-and-span-attribute-allowlist.md)
decision 3): collection, mode, counts, latency, typed failure code. Never query
text, chunk content, citation spans, absolute source paths, or metadata-filter
values. A span attribute is strictly worse than an INFO log for the same data —
it leaves the process by design.

## Kubernetes

One replica and `strategy: Recreate`, both forced by the storage model rather
than chosen: the index is a file-based store on a `ReadWriteOnce` claim, so a
second pod cannot mount it and a rolling update would hang on a Multi-Attach
error rather than fail. Every deploy is honest downtime.

The probes target `GET /v1/collections`, because there is no health endpoint to
target and adding one is a `src/` change with a real cost — ADR-0014's parity
test asserts that the registered route set is exactly the four declared
operations, and a fifth route means widening a security-relevant exclusion.
`docs/specs/phase-6-iac-observability.md` §4.2 records the tradeoff and what
the probe does and does not prove. In short: it proves the process is serving
HTTP and the index directory is listable; it does **not** prove any collection
is usable, because an empty or missing index directory returns `[]` rather than
an error.

Two one-shot manifests sit beside the base and are deliberately not part of it —
a Job's pod template is immutable, so it would break the second `apply -k`, and
a loader pod would hold the RWO claim the Deployment needs.

## Terraform (AWS)

One EC2 instance, one encrypted EBS volume with its own lifecycle, and no
inbound path.
[ADR-0020](../adr/ADR-0020-terraform-target-single-host-with-block-storage.md)
records the alternatives; the decisive one is not provider preference but
storage: SQLite's WAL coordination uses a memory-mapped `-shm` sidecar that is
not reliable over a network filesystem, which disqualifies every
serverless-container-plus-managed-filesystem answer before compute is even
considered.

The module creates **no** backup schedule. `prevent_destroy` on the data volume
means it survives instance replacement and nothing more, and SPEC.md §7 is
explicit that the SQLite store holds document text rather than a rebuildable
index. Backup scope and retention are product decisions owed before any real
deployment.

## Verification status

SPEC.md §1.4 requires each IaC path be verified with the verification date
recorded, and SPEC.md §2 forbids publishing a date no run produced. Both apply,
so the table in `infra/README.md` has filled and empty rows — the Terraform
module's `fmt`/`validate` and template rendering, the compose file's parse, and
the Kubernetes render are all checked and dated; a real `docker compose up`, a
cluster apply and a Terraform apply are not. That table is the authority and it
is updated only by someone who ran the thing.
