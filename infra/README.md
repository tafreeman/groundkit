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

The service also validates the `Host` header on both transports now
([ADR-0024](../docs/adr/ADR-0024-host-header-validation-on-both-transports.md)),
but `derive_host_allow_list` turns that check off the moment the bind is
routable, and every surface in this directory binds `0.0.0.0` deliberately —
that is the whole point of the `--allow-remote-access` flag below. So `Host`
validation does not apply to anything in `infra/`: it is not a second layer
behind the network control each surface re-establishes, it is not present at
all. Do not read the compose, Kubernetes or Terraform paths as gaining a
defense they did not have before `b096a39` — the boundary here is, and stays,
the network control in front of the bind: the loopback publish, the
`ClusterIP` + `NetworkPolicy`, the security group with no ingress rules. Only
a *host-side* `grk serve` (outside a container, on its default `127.0.0.1`
bind) enforces `Host` — see
[the deployment guide](../docs/guides/deployment.md) rather than this file for
that path.

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

**Compose** (needs a Docker daemon; nothing else, and no credentials). The first
command waits on Ollama's healthcheck rather than merely on its container, so a
cold stack does not race the daemon's listener:

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

# Delete first, always. A Job's pod template is immutable, so `apply` over a
# completed one creates no pod and changes nothing — and the `wait` below then
# returns instantly on the OLD completion. Within the hour before its TTL
# collects it, re-running the ingest without this line silently ingests nothing.
kubectl -n groundkit delete job/groundkit-ingest --ignore-not-found
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
make an egress-free subnet work. Setting `embedding_base_url` adds an egress rule
for that endpoint derived from the URL; a DNS-name host on a non-443 port also
needs `embedding_egress_cidr`, and says so at `plan` rather than at request time.

## Traces: the instrumentation has landed, and a compose run has proved it

Phase 6 lands in two changes (phase spec §3). Change 1 was the additive half:
`infra/`, the spec, the ADRs, the docs and the CI gate, with no change under
`src/`. Change 2 has now landed the instrumentation — `telemetry.py`, spans on
`Indexer.index_source`/`Indexer.index_directory`, `Retriever.search` and
`Synthesizer.synthesize`, and the JSON log formatter (ADR-0022). The collector
and Jaeger have been real, running and correctly wired since change 1, and
groundkit should now emit spans wherever a live stack exercises those three
call paths.

**That was run on 2026-08-16**, and the trace rows in the table below carry
that date: the collector→Jaeger leg on its own via a synthetic OTLP/HTTP
payload, then real `ingest` and `retrieve` spans from groundkit itself, and —
in a later run the same day — the `synthesize` span, which completes SPEC.md
§3's three-site list. The "Verification status" table is the record of what has
and has not been run, and this section is not a substitute for it: read the
scope notes under the table before treating any row as broader than it is. In
particular the `ingest`/`retrieve` run did **not** use the compose `groundkit`
service or its loopback publish, and the `synthesize` run reached its chat
model on the host rather than in the stack.

Two things are worth keeping from before that run, because they are still how
you debug this stack. The collector→Jaeger leg is provable independently by
POSTing a synthetic OTLP/HTTP payload to the collector's `:4318/v1/traces` and
finding it in the UI; and `docker compose logs otel-collector` shows the
`debug` exporter's view, which separates a receive problem from an export one.

**If you get no spans at all, check this first.** Installing the `otel` extra
and setting the `OTEL_*` variables is *not* enough on its own — those are read
by `opentelemetry.sdk._configuration` under the `opentelemetry-instrument`
launcher, not on import, so without `telemetry.configure_tracing()` having run
the API hands out non-recording proxy spans with no error and no warning. That
is precisely how the first version of this instrumentation passed its whole
unit suite while exporting nothing (ADR-0022 decision 1 carries the amendment).
`grk` calls it from its entry point, so a container running `grk` is fine; a
process importing groundkit as a library and never calling it is not.

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
| compose | `up`, ingest, a real search over the loopback publish | **passed 2026-08-16** — the documented cold-start sequence in full: `--profile setup run --rm ollama-pull`, `--profile ingest run --rm ingest` (43 files, 1299 chunks, 1299 vectors via Ollama), `up -d`, then `GET /v1/collections` → `["default"]` and `POST /v1/search` → 200 with a citation-bearing result, both over `127.0.0.1:8765` |
| compose | published ports are loopback-only | **passed 2026-08-16, host-side only** — a differential through this host's own LAN interface address, not loopback: `10.0.0.16:8766` (a control container published `0.0.0.0`) returned 200 while `10.0.0.16:8765` (the `groundkit` service, published `127.0.0.1`) was actively refused, with `Get-NetTCPConnection` showing `0.0.0.0` and `127.0.0.1` respectively. **The from-another-host leg was attempted and did NOT complete** — see the scope note below |
| compose | collector→Jaeger leg, proved on its own with a synthetic OTLP/HTTP payload | **passed 2026-08-16** |
| compose | a groundkit span visible in Jaeger, carrying no query text | **passed 2026-08-16** — `ingest` and `retrieve` spans observed; see the note below for what this run did and did not cover |
| compose | the `synthesize` span in a real trace, allowlist holding | **passed 2026-08-16** — observed from the compose `groundkit` image against the running stack; see the scope note below |
| k8s | `kubectl kustomize` renders; every manifest parses as YAML | **passed 2026-08-15** |
| k8s | `apply -k`, Job completes, Deployment Ready, port-forward serves | **passed 2026-08-16 on a SINGLE-NODE cluster** — Docker Desktop 4.86.0 Kubernetes, kind mode, server v1.36.1, 1 node. The documented sequence run verbatim including the scale-down steps: `apply -k`, corpus loader + `kubectl cp` (43 files), ingest Job completed (43 files, 1299 chunks, BM25-only — the k8s stack has no Ollama), Deployment reached 1/1 Ready against the `GET /v1/collections` probe, and `port-forward` served a citation-bearing `POST /v1/search`. **The multi-node `ReadWriteOnce` path was NOT exercised** — see the note below |
| terraform | `fmt -check`, `validate` on providers 5.100.0 and 6.60.0 | **passed 2026-08-16** |
| terraform | `bootstrap.sh.tftpl` renders; rendered script passes `bash -n` | **passed 2026-08-16** |
| terraform | ECR, egress and instance-architecture inputs classify correctly | **passed 2026-08-16** |
| compose | the one-shots gate on Ollama's healthcheck, `serve` deliberately does not | **passed 2026-08-16** |
| terraform | the security-group preconditions actually reject | **passed 2026-08-16** — `terraform test` with `mock_provider`, 5 runs, no credentials and no API call |
| terraform | `plan` / `apply` against a real account | **passed 2026-08-16** — real AWS account, `us-east-1`, single-node personal sandbox; see the note below for exact scope |

What the passing rows do **not** cover, so a full column of green is not read as
more than it is: no manifest has reached a cluster, and `terraform validate`
makes no API call — a missing IAM permission, an instance type unavailable in
the region, or an AMI filter matching nothing are all invisible to it. The
`plan` / `apply` row below is the exception: it is the one row that did make
real API calls, in one specific region and account, and its own scope note
says exactly what that does and does not settle.

**Scope of the 2026-08-16 Kubernetes run, and why "single-node" is written into
the row rather than left to be inferred.** This file already warned that the
scale-down steps are unnecessary on a single-node cluster and that omitting
them passes in the small case and stalls in the real one. The run that closed
this row *was* the small case. The steps were followed verbatim anyway, so the
documented sequence is confirmed self-consistent — but a single node can never
produce the multi-attach failure the `ReadWriteOnce` claim guards against,
because there is no second node for the volume to be attached from. **The
multi-node behaviour of this manifest set is still unverified**, and that is
the one thing a green row here must not be read as covering.

What the run did establish, none of which needs a second node: the manifests
apply as a set, the PVC binds, the ingest Job runs to completion against it,
the Deployment reaches Ready on a probe that is a real operation rather than a
static handler, the NetworkPolicy and `ClusterIP` are in place, the pod runs
with `readOnlyRootFilesystem`, all capabilities dropped and no privilege
escalation, and a citation-bearing search is served over `kubectl port-forward`.

**A gotcha worth recording for the next person, because it is not in the
sequence above.** Docker Desktop's Kubernetes runs in **kind mode**, whose
containerd image store is *separate from the host Docker daemon's*. A local
`groundkit` build is therefore invisible to the cluster and
`imagePullPolicy: IfNotPresent` still ends in `ImagePullBackOff` — the same
symptom the unpublished-placeholder trap produces, from an unrelated cause.
Loading it explicitly is what fixes it:

```bash
docker tag groundkit:local ghcr.io/tafreeman/groundkit:dev
docker save ghcr.io/tafreeman/groundkit:dev -o gk.tar
docker exec -i desktop-control-plane ctr -n k8s.io images import - < gk.tar
```

`busybox:1.37` needs the same treatment for the corpus-loader pod.

**Scope of the 2026-08-16 loopback-publish check, because "passed, host-side
only" is a real qualifier and not a hedge.** ADR-0021 decision 1's claim is
that containerising groundkit does not publish it beyond loopback. What was
demonstrated: a control container published on `0.0.0.0:8766` answered on
`10.0.0.16:8766` (this host's LAN interface address, not `127.0.0.1`), while
the `groundkit` service published on `127.0.0.1:8765` was actively refused on
`10.0.0.16:8765` in the same minute. Two ports, one host, the same Docker
publish mechanism, differing only in bind address — and the refusal is the
kernel declining a connection to an address nothing is bound to, which is the
same path a remote packet meets at its destination.

What that does **not** cover, and why the row says host-side only: the packets
never crossed the network, so firewall rules, routing and AP behaviour were
not exercised. A genuine from-another-host check was attempted the same day
from a phone and **could not complete** — that phone could not reach the
`0.0.0.0` control port either, though this host reaches it fine over the same
address, which places the fault in the wireless network rather than in
anything here. The Wi-Fi is a `-Guest` SSID with client isolation, so no device
on it can reach this machine at all. Closing the remaining leg needs a wired
host, or a device on the non-guest network. Recorded rather than quietly
counted as done, because "the check passed" and "the check could not run" are
opposite claims and a green row must not blur them.

Groundkit's own access log corroborates the negative independently: across the
whole session it recorded requests from `127.0.0.1` and from Docker's bridge
gateway `172.18.0.1` (the healthcheck) and **from no `10.0.0.0/24` address at
all**.

**Scope of the 2026-08-16 trace verification, stated because the row above is
narrower than it looks.** The collector and Jaeger were started with
`docker compose up -d otel-collector jaeger`. groundkit itself then ran as
`docker run` containers attached to that stack's network (`groundkit_default`)
performing a real `ingest` and a real `search`, with the standard `OTEL_*`
variables set. Observed in Jaeger: `groundkit.ingest.index_directory` and
`groundkit.retrieve.search`, the latter carrying `collection`, `retrieval.mode`,
`retrieval.stage`, `top_k`, `result_count` and `duration_ms` — and a sweep of
the raw trace payload for the search's own query string, the corpus path, and
the document's text found none of them. Three things that run did **not** do,
each of which is why a separate row above is still unfilled:

- It did not bring the `groundkit` service up under compose or reach it through
  the host-loopback publish, so the `up`, ingest, a real search over the
  loopback publish row is untouched — and neither is the LAN bind check, which
  is the actual security claim (ADR-0021 decision 1).
- It ran BM25-only, so no embedding-identity attributes were exercised.

**Scope of the 2026-08-16 `synthesize` span verification** (the row added
above; it was a separate, later run). The full four-service stack was up
(`groundkit`, `ollama`, `otel-collector`, `jaeger`). A one-off
`docker compose run --rm --no-deps groundkit answer …` against `/data/index`
returned a cited answer over 2 BM25 results, and
`groundkit.synthesize.synthesize` appeared in Jaeger carrying exactly
`groundkit.chat.model`, `groundkit.chat.provider`, `groundkit.duration_ms` and
`groundkit.result_count`. A sweep of the exported payload for the question
text, the completion text, both citations' offsets, the corpus path and the
source filename found **none of them** — ADR-0022 decision 3's allowlist holds
on the span it matters most on.

Three caveats, so the row is not read as more than it is:

- **The chat model was the operator's host Ollama** (`host.docker.internal`),
  not the stack's own `ollama` service, whose volume holds only the embedding
  model. `OllamaChat._allow_private_endpoint` is what permits that address.
  Nothing about the span path differs, but the stack was not self-contained
  for this run.
- The run used `--chat-model qwen3:1.7b`. The first attempts used a larger
  local model and **timed out against `ChatConfig.timeout_seconds`' 60-second
  default**, which there is no CLI flag to raise.
- Those timeouts did produce a useful negative result worth keeping: the
  **error-path** span carried `otel.status_code=ERROR` and
  `groundkit.failure_kind=ChatError` and still leaked nothing — the allowlist
  holds where a completion never arrived, which is where a leak would be
  easiest to miss.

**This row was earned twice, and the first attempt is worth recording**, since
it is the failure a green unit-test suite cannot see. The first run exported
*nothing*: the SDK was installed and every `OTEL_*` variable was set, but
`opentelemetry-api` still handed out a `ProxyTracerProvider` because those
variables are read by `opentelemetry.sdk._configuration` under the
`opentelemetry-instrument` launcher, not on import — so with no explicit
`set_tracer_provider` call, every span was non-recording, with no error and no
warning anywhere. `telemetry.configure_tracing()` is the fix, and
`tests/test_telemetry.py::TestConfigureTracing` is the regression test. Nothing
short of running the stack would have caught it.

**Scope of the 2026-08-16 `plan`/`apply` verification.** Ran against a real AWS
account (single personal free-tier sandbox, not a production or shared
account), region `us-east-1`, the account's default VPC.

The default VPC's six subnets are all public (`MapPublicIpOnLaunch: true`, a
route table with only an internet-gateway route) and the account had no NAT
gateway anywhere in the region. That matters because of the "Network
prerequisite" section above: the module pins `associate_public_ip_address =
false` unconditionally, so an instance launched into any of those subnets
as-is would get no public IP and no usable route out through the IGW —
`dnf install -y docker` would fail under `set -e` during bootstrap, the exact
"healthy instance, no service" trap this file already warns about. A NAT
gateway, a new route table (`0.0.0.0/0` → NAT), and its association with one
subnet were created **as a prerequisite outside the module** before `plan`,
and destroyed afterward along with everything else — the module itself does
not build a NAT gateway (ADR-0020 decision 3) and this run did not change
that.

`container_image` was a real private ECR reference
(`<account>.dkr.ecr.us-east-1.amazonaws.com/groundkit:phase6-verify`), pushed
from the locally-built `groundkit:local` image for this run. `plan` showed 9
resources to add, 0 to change, 0 to destroy, with a real AMI resolved
(`al2023-ami-2023.*-x86_64` in `us-east-1`) and the ECR-detection logic
correctly matching the image reference and attaching
`AmazonEC2ContainerRegistryReadOnly`. `apply` created all 9 cleanly. On the
instance: bootstrap installed docker, authenticated to the private ECR
registry and pulled the image, formatted and mounted the data volume via the
retrying `groundkit-storage.service` unit, and `groundkit.service` came up
healthy — proving the ECR pull path and the storage-prep unit for real, not
just at `bash -n`/`terraform console`. A document was placed under
`/srv/groundkit/corpus`, `groundkit-ingest` was run, and `groundkit` was
restarted; a real SSM port-forward session was then opened from the
operator's own machine and a real `POST /v1/search` over that tunnel returned
a correct, citation-bearing result for a planted marker token — that, not the
successful `apply`, is what actually closes this row. `terraform destroy` ran
in the same session; the ECR repository, NAT gateway, route table and Elastic
IP (none of them terraform-managed) were deleted immediately afterward.

What this run did **not** cover: the dense/hybrid path (`embedding_base_url`
stayed unset, so the derived-egress-rule branch was exercised only by the
existing `terraform console` checks, never against a real security group);
`create_ssm_vpc_endpoints` (stayed `false` — this account had NAT-based
egress, so the SSM-interface-endpoint resources were never applied for real);
and any partition other than the commercial one (GovCloud/China ECR-host and
managed-policy-ARN matching remain `terraform console`-only, as before). It
is also a single run against a single account and region — the same caveat
the k8s single-node row already carries.

Update the rows, with the date, when you run one. Do not update a row you did
not run.
