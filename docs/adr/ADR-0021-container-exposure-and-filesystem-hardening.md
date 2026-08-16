# ADR-0021 — In a container the loopback guarantee moves outward, and the read-only claim splits in two

- **Status:** Accepted (owner, 2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

Two Phase 4 properties do not survive containerisation unchanged, and both are
load-bearing enough that leaving the change implicit would quietly weaken a security
claim the repo makes in writing.

**The bind guard cannot hold inside a container.** ADR-0014 decision 7 makes the
`127.0.0.1` bind the service's only access control, and `service/binding.py` enforces it:
a non-loopback `--host` is refused unless `--allow-remote-access` is passed, which then
logs a warning naming exactly what is published. Inside a container, a process bound to
`127.0.0.1` is reachable only from that container's own network namespace — not from the
host, not from a sibling container, not through a published port. A containerised
groundkit that keeps the process-level guarantee serves nobody. The guarantee therefore
has to move somewhere, and *where* it moves is a decision rather than an implementation
detail.

**The read-only claim was already qualified, and Phase 6 is where the qualification is
cashed.** KNOWN_LIMITATIONS.md records that read-only "does not mean the process writes
no bytes": opening a WAL database updates its `-shm`/`-wal` sidecars, and
`SQLiteMetadataStore.open` runs `CREATE TABLE IF NOT EXISTS` and a best-effort chmod on
every open (ADR-0014 decision 4). The entry ends by deferring filesystem-level
enforcement — a read-only mount — to this phase. Cashing that deferral means finding out
which mounts can actually be read-only, and the answer is not "all of them."

## Decision

### 1. The container binds `0.0.0.0`; the loopback guarantee is re-established at the publish boundary

The image's default command passes `--host 0.0.0.0 --allow-remote-access`. Within the
container's namespace that is the only address that can serve a published port, and the
acknowledgement flag is passed because the guard is correct to demand it: something *is*
being exposed, and the operator's warning line in the log is accurate.

What replaces the process-level guarantee is the mapping one layer out, and each
deployment surface states it explicitly:

- **compose** publishes `127.0.0.1:8765:8765` — a host-loopback publish, so the socket
  the outside world could reach does not exist.
- **Kubernetes** uses a `ClusterIP` Service **plus a default-deny-ingress
  NetworkPolicy**, and it takes both. `ClusterIP` closes the cluster's *edge* — not
  `NodePort`, not `LoadBalancer` — but every pod in every namespace can dial a ClusterIP
  directly, so on its own it is a smaller blast radius rather than a boundary. This
  record originally claimed the Service alone re-established the guarantee; it does not,
  and `infra/k8s/networkpolicy.yaml` is the object that does. Reaching the service is
  `kubectl port-forward`, authorised by the cluster's own RBAC.
- **Terraform** (ADR-0020 decision 2) publishes `127.0.0.1:8765` on the instance and
  opens no ingress at all; access is SSM port forwarding.

The Kubernetes row is the weakest of the three and is worth saying so plainly: a
NetworkPolicy is **silently inert** on a cluster whose CNI does not enforce one — the
API server accepts it, reports no status and emits no warning — so that surface's
guarantee is contingent on a cluster property this repo cannot check from a manifest.
compose and Terraform both rest on a kernel-level socket bind instead, which is not
contingent on anything.

**The consequence, stated plainly because it is the cost of decision 1: the image is not
safe to run with a bare `-p 8765:8765` or with `--network host`.** Either publishes an
unauthenticated, content-bearing surface on every interface of the host. The flag inside
the image cannot distinguish the two cases — it sees `0.0.0.0` either way — so the guard
that used to be a refusal is now a warning, and the manifests are what make it true.
That is a real reduction in enforcement, and it is recorded here rather than discovered
by whoever first types the shorter command.

### 2. The root filesystem is read-only; the corpus mount is read-only; the index mount cannot be

Three mounts, three different answers, and the differences are the substance of this
decision:

| Path | Mode | Why |
|---|---|---|
| `/` (image) | read-only | Nothing the service does writes into the image. |
| `/data/corpus` | **read-only** | Citation resolution re-reads source files and never writes one. This is the filesystem-level enforcement KNOWN_LIMITATIONS.md deferred to Phase 6. |
| `/data/index` | **read-write** | SQLite WAL sidecars, `CREATE TABLE IF NOT EXISTS`, and the open-time chmod (ADR-0014 decision 4). A read-only mount here does not harden the service, it stops it starting. |

So the deferred item is discharged **in part, and the part matters**: the documents whose
spans every citation resolves against are now unwritable by the serving process, enforced
by the kernel rather than by Python. The index directory stays writable, and no
arrangement of mount flags changes that while ADR-0002 keeps SQLite in WAL mode. The
honest summary — the one that belongs in KNOWN_LIMITATIONS.md rather than a claim that
Phase 6 "made the service read-only" — is that read-only is now true of the corpus and
still false of the index.

A read-only root filesystem also requires naming every scratch path, since a missing one
surfaces as a permission error at some later moment rather than at startup. There are
exactly two, and both are `tmpfs`/`emptyDir`: `/tmp` and the service user's home
directory. If a third ever appears, the deployment fails loudly, which is the intended
behaviour and the reason the list is short and explicit rather than replaced by a
writable root.

### 3. The image runs as a fixed non-root uid, 10001, and the number is part of the interface

`groundkit:groundkit`, uid and gid both 10001, created in the image and set with `USER`.
The number is fixed rather than left to the distribution's next-free allocation because
three things outside the image have to name the same identity: a Kubernetes
`runAsUser`/`fsGroup`, the ownership Docker copies onto a fresh named volume from the
image's mountpoint, and the ownership a host bind mount must carry. A uid that moves
between rebuilds turns each of those into a silent permission failure on a volume that
was writable yesterday.

`/data/index` and `/data/corpus` are created in the image and chowned to 10001, which is
what makes a fresh Docker named volume inherit the right ownership. Kubernetes uses
`fsGroup: 10001` instead, because a PVC's ownership is the CSI driver's business rather
than the image's.

Alongside it: `allowPrivilegeEscalation: false`, all capabilities dropped, and
`seccompProfile: RuntimeDefault` on the pod. The service opens files and a socket; it
needs none of what those grant.

### 4. The image carries the `dense` extra and not `rerank`

`dense` is in, because the compose topology includes Ollama and a stack whose embedding
provider is present but whose vector store is missing would be a topology that cannot do
the thing it is arranged to do. `rerank` is out: it pulls torch, which is a
multi-gigabyte layer for a capability `Retriever.search` cannot even reach today
(ADR-0012 decision 2 keeps rerank out of `retrieval/search.py`; `grk serve --rerank`
loads a model only for a per-request flag). The same command/option line ADR-0015 drew
decides it — an image is an install, and this install carries the options the deployment
actually exercises.

## Alternatives considered

- **Keep the loopback bind inside the container and reach the service another way.**
  There is no other way: a published port, a compose-network sibling and a
  `kubectl port-forward` all arrive on the container's external interface, not its
  loopback. The alternative is a container that cannot serve.
- **Bind `0.0.0.0` without `--allow-remote-access`, by adding a "containerised"
  escape.** Rejected. Detecting containerisation is heuristic (`/.dockerenv`, cgroup
  paths, `KUBERNETES_SERVICE_HOST`) and each heuristic is spoofable by exactly the
  environment it is trying to classify. Worse, it would suppress the warning line in the
  one deployment shape where the exposure is real, trading an accurate warning for a
  tidier log.
- **Mount `/data/index` read-only and accept a degraded mode.** Rejected because there is
  no degraded mode to accept: `SQLiteMetadataStore.open` writes on every open, so the
  service does not start rather than starting with reduced function. Making it start
  would mean a read-only open path in `index/metadata.py` — genuinely interesting, a
  real src/ change, and not something to smuggle in behind a mount flag. Recorded in the
  phase spec as a named future decision instead.
- **A writable root filesystem, with hardening left to the non-root user.** Rejected:
  the two scratch paths are known and small, so the cost of naming them is one line each,
  and a read-only root is the difference between "the process has no reason to write
  there" and "the process cannot."
- **`USER groundkit` by name rather than `USER 10001`.** The image does both — the user
  exists by name and `USER` names the numeric id — because Kubernetes'
  `runAsNonRoot` check reads the numeric id from the image config and cannot resolve a
  name against the image's `/etc/passwd`. A name-only `USER` makes a pod with
  `runAsNonRoot: true` fail to start with an error about being unable to verify the user.
- **Ship `rerank` in the image so the deployment is feature-complete.** Rejected: several
  gigabytes for a flag that reaches no code path in `retrieval/search.py`, and it would
  put torch inside the artifact this repo's CI is otherwise careful never to pull.

## Consequences

- **`docker run -p 8765:8765 groundkit` publishes the corpus on every interface.** The
  image cannot prevent it. Every document that references running the image references
  the loopback publish, and the image's own `HEALTHCHECK` and default command are written
  against it.
- Filesystem-level read-only enforcement now exists for the corpus and not for the index.
  KNOWN_LIMITATIONS.md is updated to say exactly that, replacing the blanket deferral to
  this phase — a partial discharge stated precisely, rather than a checkbox.
- The uid `10001` is now part of the image's public contract. Changing it breaks existing
  volumes, so it changes with a major version and a migration note, not opportunistically.
- A read-only root filesystem will surface any future write to an unexpected path as a
  startup or runtime permission error rather than as a silent write into a container
  layer that vanishes on restart. That is the intended trade: a loud failure in a
  deployment beats an invisible one.
- The image is larger than a BM25-only install by lancedb and its dependencies, and
  smaller than a full-featured one by torch.

## References

- SPEC.md **§3** (observability and structured logging conventions; the gate set), §7
  (SQLite is content-bearing; the 127.0.0.1 bind), §9 (Phase 6 row: multi-stage non-root
  Dockerfile, compose, k8s with probes).
- [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md) —
  decision 4 (read-only is scoped to durable state; the process holds a read-write
  handle) and decision 7 (the bind is the only access control). This record is what
  happens to both when the process is containerised.
- [ADR-0002](ADR-0002-index-persistence.md) — WAL mode, which is why `/data/index` cannot
  be a read-only mount.
- [ADR-0015](ADR-0015-service-dependencies-are-base-not-an-extra.md) — the
  command/option line decision 4 applies to the image's extras.
- [ADR-0020](ADR-0020-terraform-target-single-host-with-block-storage.md) — the cloud
  deployment's own answer to decision 1, and the reason its security group has no ingress.
- `KNOWN_LIMITATIONS.md` — the "read-only does not mean the process writes no bytes"
  entry, whose deferral to Phase 6 decision 2 discharges in part.
