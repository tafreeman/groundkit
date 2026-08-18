# Deployment

This page is for anyone deciding whether, and how, to run groundkit as a
shared service other people or programs can reach — not as a command-line
tool one person runs on their own laptop. If that's not what you're doing,
you can skip this page entirely; nothing here is required for normal use.
If it is, the one fact worth taking away even from a fast skim is in the very
next section: **this service has no login and no password of any kind**, so
whatever network address it's reachable at is the *only* thing standing
between your documents and anyone who finds that address. By the end of this
page you'll know which of the deployment options fits your situation, and
exactly what each one does — and does not — do to keep that address private.

groundkit's deployment paths live in `infra/` — a container image (a
portable, self-contained package the service runs from, the same way on any
machine that can run it), a compose stack (several containers started and
networked together with one command, via Docker Compose), Kubernetes
manifests (configuration for running those containers across a cluster of
machines), and a Terraform module (code that provisions real cloud
infrastructure — here, one AWS server — the same way every time it runs).
This page is the map and the warnings; `infra/README.md` in the repository
is the operating manual, and the reasoning behind each shape is in the ADRs
linked below.

None of this is required to use groundkit. `grk ingest`, `grk search`,
`grk serve` and `grk serve-mcp` run on a laptop with no container runtime and no
credentials, which is what SPEC.md §10 makes the definition of done.

## Read this first: there is no authentication

Every operation the service exposes is read-only, which satisfies SPEC.md §7's
shared-secret requirement vacuously (trivially — the requirement only applies
to operations that change something, and there are none) — the set of
mutating operations is empty, so the set requiring a secret is empty. The
consequence is stated in
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
| Kubernetes | `ClusterIP` Service **and** a default-deny-ingress NetworkPolicy | `kubectl port-forward` |
| Terraform (AWS) | security group with no ingress rules at all | SSM port forwarding |

The Kubernetes row needs both objects, and it is the weakest of the three.
A `ClusterIP` (the default, cluster-internal-only way of exposing a set of
pods — as opposed to `NodePort` or `LoadBalancer`, which do expose one)
closes the cluster's *edge*, but any pod in any namespace can dial a
ClusterIP directly, so on its own it is a smaller blast radius rather than a
boundary. The NetworkPolicy (a Kubernetes rule restricting which pods may
talk to which) is what closes that, and a NetworkPolicy is **silently
inert** on a cluster whose CNI (the plugin that actually implements pod
networking) does not enforce one: the API server accepts it, reports no
status, and emits no warning. compose and Terraform rest on a kernel-level
socket bind instead — the operating system itself refusing the connection —
which is contingent on nothing.

The image itself cannot enforce any of this. `docker run -p 8765:8765
groundkit` — the obvious shorter command — publishes an unauthenticated,
content-bearing surface on every interface of the host, and so does
`--network host`.

## The container image

This is what every option below actually runs. It is multi-stage (built in
stages so the shipped image contains only what's needed to run, not the
tools used to build it), built from the repository root, running as a fixed
non-root uid (`10001` — an ordinary, unprivileged user account, not the
all-powerful root) with a read-only root filesystem (the running container
cannot modify its own program files, even if something inside it is
compromised). The uid is part of the image's public contract rather than an
implementation detail: a Kubernetes `runAsUser`, a Docker named volume's
inherited ownership, and a host bind mount all have to name the same
identity.

Three mounts, three different answers, and the differences are the point:

| Path | Mode | Why |
|---|---|---|
| `/` | read-only | Nothing the service does writes into the image. |
| `/data/corpus` | **read-only** | Citation resolution (re-checking a search result against its source) re-reads source files and never writes one. |
| `/data/index` | **read-write** | WAL sidecars (extra files SQLite's write-ahead-log journaling mode keeps next to the database itself), `CREATE TABLE IF NOT EXISTS`, and the open-time chmod. |

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

One detail is easy to misread as a hole in the SSRF guard (the check that
stops groundkit from being tricked into fetching a network address it
shouldn't, such as another internal service). The service reaches Ollama at
`http://ollama:11434`, which resolves to an RFC1918 bridge address (a
private, non-internet-routable address — the same family as the ones a home
router hands out), not to loopback. That works because `OllamaEmbedder` sets
`_allow_private_endpoint` as a **class attribute** scoped to that one provider
(ADR-0014 decision 10) — not because the guard is relaxed. An OpenAI-compatible
endpoint pointed at a private address is still refused.

### Traces: instrumented and verified

Phase 6 landed in two changes. The first was purely additive infrastructure and
changed nothing under `src/`; the second added the OpenTelemetry instrumentation
(tracing — recording how long each step of a request takes and where it went,
for later inspection in a tool like Jaeger) and the JSON log formatter. Both
have landed: the collector and Jaeger are real and correctly wired, and
groundkit emits spans (individual timed records of one operation — the basic
unit a trace is built from) on `Indexer.index_source`/`index_directory`,
`Retriever.search`, and `Synthesizer.synthesize`. **If the Jaeger UI is empty
against a running stack, treat that as a fault to investigate, not as an
expected state** — `infra/README.md`'s dated "Verification status" table
records exactly what was run and when, and its "Traces" section walks the
collector→Jaeger synthetic-payload check that isolates a receive problem from
an export one.

When instrumentation does land, what a span may carry is an allowlist rather
than a denylist
([ADR-0022](../adr/ADR-0022-observability-dependency-shape-and-span-attribute-allowlist.md)
decision 3): collection, mode, counts, latency, typed failure code. Never query
text, chunk content, citation spans, absolute source paths, or metadata-filter
values. A span attribute is strictly worse than an INFO log for the same data —
it leaves the process by design.

## Kubernetes

One replica and `strategy: Recreate` (stop the old pod fully before starting
the new one, rather than running both briefly side by side), both forced by
the storage model rather than chosen: the index is a file-based store on a
`ReadWriteOnce` claim (a storage volume only one pod can mount for writing
at a time), so a second pod cannot mount it and a rolling update would hang
on a Multi-Attach error rather than fail. Every deploy is honest downtime.

The probes (Kubernetes' own automated health checks, used to decide whether
to send traffic to a pod or restart it) target `GET /v1/collections`,
because there is no health endpoint to target and adding one is a `src/`
change with a real cost — ADR-0014's parity test asserts that the registered
route set is exactly the four declared operations, and a fifth route means
widening a security-relevant exclusion.
`docs/specs/phase-6-iac-observability.md` §4.2 records the tradeoff and what
the probe does and does not prove. In short: it proves the process is serving
HTTP and the index directory is listable; it does **not** prove any collection
is usable, because an empty or missing index directory returns `[]` rather than
an error.

Two one-shot manifests sit beside the base and are deliberately not part of it —
a Job's pod template is immutable, so it would break the second `apply -k`, and
a loader pod would hold the RWO claim the Deployment needs. The ingest Job has
its own kustomization rather than being a loose `kubectl apply -f`, because a
kustomize `images:` transformer rewrites only its own kustomization's resources:
applied loose, the Job kept the unpublished placeholder image and the documented
workflow ended in `ImagePullBackOff`. Set the image in both places.

**Delete the ingest Job before re-applying it.** A Job's pod template is
immutable, so `apply` over a completed one is accepted, creates no pod, and
leaves a following `wait --for=condition=complete` to return immediately on the
previous run's completion. Inside the hour before `ttlSecondsAfterFinished`
collects it, re-ingesting after copying new documents therefore reports success
having done nothing.

**Scale the Deployment to zero before running either one-shot.** One RWO claim
admits one pod; on a multi-node cluster the one-shot otherwise sits Pending on a
multi-attach error. Scaling back up afterwards is not just restoring service —
a `Retriever` is a snapshot as of `open()` and never refreshes, so the restart
*is* the reopen that makes newly ingested content visible. On a single-node
cluster the scale-down is unnecessary, which is exactly what makes skipping it a
trap: it passes in the small case and stalls in the real one.

## Terraform (AWS)

One EC2 instance (a single virtual machine rented from AWS), one encrypted
EBS volume (its attached block-storage disk, AWS's equivalent of a hard
drive) with its own lifecycle, and no inbound path.
[ADR-0020](../adr/ADR-0020-terraform-target-single-host-with-block-storage.md)
records the alternatives; the decisive one is not provider preference but
storage: SQLite's WAL coordination uses a memory-mapped `-shm` sidecar that is
not reliable over a network filesystem, which disqualifies every
serverless-container-plus-managed-filesystem answer before compute is even
considered.

The rest of this section is Terraform-specific operational detail for
whoever actually runs `terraform apply` — skip to
[Verification status](#verification-status) below if you only need the
shape of the guarantee. Five operational facts that are easy to get wrong
and fail silently. They share a shape worth naming: each one applies, boots
and passes a probe, then fails at a distance from its cause.

- **The instance needs outbound HTTPS during bootstrap** — it installs docker
  and pulls the image — so a private subnet wants a NAT gateway.
  `create_ssm_vpc_endpoints` carries the Session Manager control channel only;
  on an otherwise egress-free subnet it yields an instance you can open a
  session to and no service running on it.
- **The dense path needs its own egress rule, and the module derives one.**
  Ollama listens on 11434 and the standing egress rule is 443, so
  `embedding_base_url` also produces an egress rule scoped to that endpoint. A
  DNS-name host on a non-443 port cannot be resolved to a CIDR at plan time and
  needs `embedding_egress_cidr`; an IPv6 endpoint is refused at any port, since
  every rule the module writes is `cidr_ipv4` and "443 is already covered" is
  true of IPv4 alone. Both fail at `plan` with the reason, which is the whole
  point — the alternative is an embed call that times out on a deployment that
  applied cleanly.
- **The deployment is x86_64 in two places, so `instance_type` is validated.**
  The AMI filter selects an x86_64 image and the container image is built for a
  single architecture, so a Graviton type is refused at `plan` rather than by
  EC2 at launch. Supporting arm64 is a multi-architecture image build, not a
  different value for that variable.
- **A private ECR image is detected and authenticated.** When
  `container_image` matches the private-ECR host pattern (including
  `.amazonaws.com.cn`), the role gains `AmazonEC2ContainerRegistryReadOnly` and
  bootstrap performs a `docker login`; otherwise neither happens. Detected
  rather than flagged because an unauthenticated pull of a private image ends
  bootstrap before the service unit is written, and the instance then looks
  healthy while serving nothing.
- **The service requires its data mount, not merely ordering after it.** The
  fstab entry uses `nofail` on purpose — there is no SSH ingress, so an instance
  stuck in emergency mode is unreachable — which means a reboot without the EBS
  volume still boots. `RequiresMountsFor=/srv/groundkit` on the unit is what
  stops docker from creating the bind-mount paths on the root disk and serving
  an empty index that looks healthy, with any ingest into it shadowed the moment
  the volume returns.

**The module creates no backup schedule.** `prevent_destroy` on the data
volume means it survives instance replacement and nothing more, and
SPEC.md §7 is explicit that a collection's data is content, not a
rebuildable index: the SQLite store, its `.lance` vector-store sibling, and
its `.snapshots` directory of fetched URL content all hold document text,
and none of the three is inferable from another's absence — a backup that
copies the SQLite file alone cannot restore what a `.snapshots`-backed
citation needs to verify against. Backup scope and retention are product
decisions owed before any real deployment.

## Verification status

SPEC.md §1.4 requires each IaC path be verified with the verification date
recorded, and SPEC.md §2 forbids publishing a date no run produced. Both apply,
and the table in `infra/README.md` now carries a dated row for every path
listed above — the Terraform module's `fmt`/`validate` and template rendering,
the compose file's parse, a real `docker compose up` with an ingested search
served over the loopback publish, a Kubernetes `apply -k` reaching Ready, and a
Terraform `plan`/`apply` against a real AWS account. Several rows carry a scope
note rather than a blanket pass — the multi-node `ReadWriteOnce` path, for one,
is explicitly called out as still unverified — so read those notes before
treating a green row as broader than it is. That table is the authority and it
is updated only by someone who ran the thing.

## Next

[MCP clients](mcp-clients.md) covers connecting an AI assistant to a server
you've stood up — read this page's "no authentication" warning first if that
server will be reachable by anything other than you. [Security](../security.md)
and [Known limitations](../limitations.md) cover what is and is not hardened
beyond what's on this page.
