# `infra/` — deployment paths for groundkit

Created by Phase 6 and by no earlier phase: SPEC.md §3 is explicit that empty
IaC directories are decoration, so this directory arrives with its contents.

The design decisions behind everything here are in three ADRs and the phase
spec, and they are worth reading before editing any file in this tree:

- [ADR-0020](../docs/adr/ADR-0020-terraform-target-single-host-with-block-storage.md)
  — why the cloud target is one EC2 instance with a block volume and no inbound
  path.
- [ADR-0021](../docs/adr/ADR-0021-container-exposure-and-filesystem-hardening.md)
  — why the container binds `0.0.0.0`, and why `/data/corpus` is a read-only
  mount while `/data/index` cannot be.
- [ADR-0022](../docs/adr/ADR-0022-observability-dependency-shape-and-span-attribute-allowlist.md)
  — the telemetry dependency shape and the span-attribute allowlist.
- [Phase 6 spec](../docs/specs/phase-6-iac-observability.md) — the whole plan,
  including what is deferred and why.

## The one thing to know before running any of this

**Phase 4 ships no authentication of any kind.** Every operation is read-only
(ADR-0014 decision 1), which satisfies SPEC.md §7's shared-secret clause
vacuously, so the address the service is reachable at is its *entire* access
control. A `search` response carries document text and absolute source paths.

Inside a container the process must bind `0.0.0.0` or nothing can reach it, so
that guarantee moves one layer out and each surface here re-establishes it:

| Surface | What keeps it private |
|---|---|
| compose | `ports: 127.0.0.1:8765:8765` — a host-loopback publish |
| Kubernetes | a `ClusterIP` Service **and** a default-deny-ingress NetworkPolicy; reach it with `kubectl port-forward` |
| Terraform | no ingress rules at all; reach it with SSM port forwarding |

The Kubernetes row takes both objects and is the weakest of the three. `ClusterIP`
closes the cluster's *edge*; every pod in every namespace can still dial a
ClusterIP directly, so `networkpolicy.yaml` is what actually closes in-cluster
reachability — and a NetworkPolicy is **silently inert** on a cluster whose CNI
does not enforce one. Read that file's header before treating it as a guarantee.

`docker run -p 8765:8765 groundkit` — the obvious shorter command — publishes an
unauthenticated, content-bearing surface on every interface of the host. The
image cannot tell that case from the safe one.

## Layout

```
docker/Dockerfile          multi-stage, non-root (uid 10001), read-only-root ready
compose/docker-compose.yml service + Ollama + OTel collector + Jaeger
compose/otel-collector.yaml  OTLP in; OTLP to Jaeger + debug out
compose/.env.example       names only, never values
k8s/                       namespace, NetworkPolicy, PVC, Deployment, Service
k8s/ingest/                the ingest Job, its own kustomization (image override)
k8s/pod-corpus-loader.yaml a `kubectl cp` target; plain `-f`, needs no override
terraform/aws-ec2/         ADR-0020's module
```

## Quick starts

**Compose** (needs a Docker daemon; nothing else, and no credentials):

```bash
docker compose -f infra/compose/docker-compose.yml --profile setup run --rm ollama-pull
docker compose -f infra/compose/docker-compose.yml --profile ingest run --rm ingest
docker compose -f infra/compose/docker-compose.yml up -d
curl -s http://127.0.0.1:8765/v1/collections
# Jaeger UI: http://127.0.0.1:16686  (see "Traces" below before looking for spans)
```

**Kubernetes** (needs a cluster and a pushed image — none is published). Set the
image in **both** `k8s/kustomization.yaml` and `k8s/ingest/kustomization.yaml`
first; a kustomize `images:` transformer reaches only its own kustomization's
resources.

```bash
kubectl kustomize infra/k8s                      # render, offline
kubectl apply -k infra/k8s

# The PVC is ReadWriteOnce, so exactly one pod may hold it. Scale down before
# running either one-shot, or on a multi-node cluster they stall Pending on a
# multi-attach error rather than failing.
kubectl -n groundkit scale deploy/groundkit --replicas=0

kubectl apply -f infra/k8s/pod-corpus-loader.yaml
kubectl -n groundkit wait --for=condition=Ready pod/groundkit-corpus-loader
kubectl -n groundkit cp ./docs groundkit-corpus-loader:/data/corpus
kubectl -n groundkit delete pod groundkit-corpus-loader

kubectl apply -k infra/k8s/ingest
kubectl -n groundkit wait --for=condition=complete job/groundkit-ingest --timeout=10m

# Scaling back up is also the retriever reopen: a Retriever is a snapshot as of
# open() and never refreshes, so a pod that had been serving would return zero
# results for everything the ingest just added.
kubectl -n groundkit scale deploy/groundkit --replicas=1
kubectl -n groundkit port-forward svc/groundkit 8765:8765
```

On a single-node cluster the scale-down steps happen to be unnecessary, which is
precisely what makes omitting them a trap: the sequence passes in the small case
and stalls in the real one.

**Terraform**: see [`terraform/aws-ec2/README.md`](terraform/aws-ec2/README.md).
The instance needs outbound HTTPS during bootstrap — it installs docker and
pulls the image — so a private subnet wants a NAT gateway.
`create_ssm_vpc_endpoints` covers the SSM control channel only and does **not**
make an egress-free subnet work.

## Traces: the stack is wired, and nothing emits yet

Phase 6 lands in two changes (phase spec §3). This one is the additive half:
`infra/`, the spec, the ADRs, the docs and the CI gate, with **no change under
`src/`**. The collector and Jaeger are real, running and correctly wired — and
groundkit emits no spans until the instrumentation change lands, so the Jaeger
UI will show none.

The collector→Jaeger leg is provable on its own in the meantime by POSTing a
synthetic OTLP/HTTP payload to the collector's `:4318/v1/traces` and finding it
in the UI. `docker compose logs otel-collector` shows the `debug` exporter's
view, which separates a receive problem from an export one.

## Air-gap

groundkit itself makes no outbound connection except to the embedding endpoint
you configure. The stack needs a network exactly twice, both one-off:

1. pulling the four images (see the pin table below);
2. pulling the embedding model into the `ollama-models` volume — the
   `setup` profile does this, and a pre-seeded volume replaces it.

After that, `docker compose up` needs nothing outside the host. That is the
honest shape of "air-gap friendly" for a stack containing a model server: the
model has to arrive somehow.

## Pinned images

None of these tags was pulled while this tree was written — no container runtime
was available (phase spec §6.1). CI resolves each one with
`docker buildx imagetools inspect`, which reads the manifest without downloading
a layer and is what makes a pin a checked claim rather than a guess.
**All six resolved on 2026-08-15** in the first run of that job; the two base
images additionally proved themselves by building. They were chosen for
confidence that they exist, not for recency, so a resolving tag is not
necessarily a current one — refresh them at the first `compose up` and record
that date here.

| Image | Pin | Used by |
|---|---|---|
| `ghcr.io/astral-sh/uv` | `0.11.23` | Dockerfile builder — the uv release this repo's `uv.lock` was produced with |
| `python` | `3.11-slim-bookworm` | Dockerfile builder and runtime, same tag in both (see the file for why) |
| `ollama/ollama` | `0.5.4` | compose |
| `otel/opentelemetry-collector` | `0.115.0` | compose — core, not contrib |
| `jaegertracing/all-in-one` | `1.62.0` | compose |
| `busybox` | `1.37` | `k8s/pod-corpus-loader.yaml` |

## Verification status

SPEC.md §1.4 requires each IaC path be "verified to work, with the verification
date recorded." SPEC.md §2 forbids publishing a date that was not earned by a
run. Both hold here, so some rows are empty.

**No container runtime, no cluster and no cloud account were available on the
machine this tree was written on** — the Docker CLI was present but its daemon
was not running, `kubectl` had no configured context, and there was no AWS
credential. Nothing was built, pulled, applied or planned *there*. The rows
marked *(CI)* were earned by the `infra` job on a runner instead, which is why
that job exists rather than being a formality.

| Path | Check | Status |
|---|---|---|
| all | six pinned third-party tags resolve | **passed 2026-08-15** *(CI)* |
| Dockerfile | `docker build` | **passed 2026-08-15** *(CI)* |
| Dockerfile | container runs as uid 10001; CLI works; starts under `--read-only` with only the two named scratch mounts | **passed 2026-08-15** *(CI)* |
| compose | `docker compose config` parses and interpolates | **passed 2026-08-15** |
| compose | `up`, ingest, a real search over the loopback publish | **not yet run** |
| compose | a groundkit span visible in Jaeger | **blocked** — needs the instrumentation change |
| k8s | `kubectl kustomize` renders; every manifest parses as YAML | **passed 2026-08-15** |
| k8s | `apply -k`, Job completes, Deployment Ready, port-forward serves | **not yet run** |
| terraform | `fmt -check`, `validate` on providers 5.100.0 and 6.60.0 | **passed 2026-08-15** |
| terraform | `bootstrap.sh.tftpl` renders; rendered script passes `bash -n` | **passed 2026-08-15** |
| terraform | `plan` / `apply` against a real account | **not yet run** |

What the passing rows do **not** cover, so a full column of green is not read as
more than it is: no container has served a request, no manifest has reached a
cluster, and `terraform validate` makes no API call — a missing IAM permission,
an instance type unavailable in the region, or an AMI filter matching nothing
are all invisible to it.

Update the rows, with the date, when you run one. Do not update a row you did
not run.
