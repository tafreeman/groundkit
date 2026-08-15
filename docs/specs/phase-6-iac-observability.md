# Phase 6 — IaC + observability

Feature spec for SPEC.md §9 Phase 6: *multi-stage non-root Dockerfile; compose
(service+Ollama+collector+Jaeger); k8s (deployment, service, PVC, probes);
Terraform module for one concrete provider; OTel verified end-to-end in compose.*

Status: **in progress — change 1 of 2 landed, change 2 outstanding.** Nothing
here overrides SPEC.md; where this document makes a decision SPEC.md does not
already contain, it names the ADR that holds it.

`infra/` is created by this phase and by no earlier one. SPEC.md §3 is explicit
that empty IaC directories are decoration, which is why the directory arrives
with its contents rather than ahead of them.

## 1. What Phases 1–4 already provide

| Asset | State | Phase 6 use |
|---|---|---|
| `grk serve` (REST + MCP over one runtime) | shipped Phase 4 | the container's default command |
| `service/binding.py` bind guard | shipped, refuses non-loopback without acknowledgement | re-established one layer out (ADR-0021 decision 1) |
| `GET /v1/collections` | shipped, opens nothing | liveness/readiness target (§4.2) |
| `SQLiteMetadataStore` in WAL mode | shipped Phase 1, ADR-0002 | decides the storage primitive (ADR-0020 decision 1) |
| `OllamaEmbedder._allow_private_endpoint` | shipped Phase 4, ADR-0014 decision 10 | makes `http://ollama:11434` on a bridge network reachable |
| `service/api.py` access log | id, method, route, tool, status, latency, results — never query text | the record JSON formatting wraps, unchanged (ADR-0022 decision 4) |
| `service/errors.py` typed `kind` | shipped, closed mapping | the failure-code attribute and log field |

Two things are **not** seated:

- **Nothing emits a span.** There is no tracer, no exporter, no
  `opentelemetry` dependency of any kind in the tree.
- **There is no health endpoint.** The four operations in `service/tools.py`
  are the complete route set, asserted by a parity test — see §4.2, which is
  the sharpest constraint in the phase.

## 2. Done means

1. A multi-stage, non-root image that builds from the repo root and runs
   `grk serve` with no writable root filesystem.
2. A compose stack — service, Ollama, OTel collector, Jaeger — that comes up
   from `docker compose up` with no credentials and no host ports beyond
   loopback.
3. Kubernetes manifests: namespace, PVC, deployment with liveness/readiness/
   startup probes, ClusterIP service, and the one-shot ingest Job that makes
   the deployment reachable at all (an index directory the service will not
   create has to be created by something).
4. A Terraform module for one concrete provider, with the provider choice
   recorded in an ADR before the module was written.
5. OTel spans on `ingest` and `retrieve`, structured JSON logs carrying request
   id, latency, result counts and typed failure codes, and neither carrying
   document content or query text.
6. Every IaC path either **verified with the date recorded**, or explicitly
   marked **not yet verified** with the reason. SPEC.md §1.4 requires the first;
   SPEC.md §2 forbids inventing it, and §8's regression-test rule is the same
   principle applied to tests. See §6.
7. All gates green: ruff, ruff-format, `mypy --strict`, pytest, both coverage
   gates, `mkdocs build --strict`, pip-audit, gitleaks.
8. SPEC.md §9 status updated, `KNOWN_LIMITATIONS.md` updated, ADRs accepted.

## 3. Two changes, not one

Phase 5 is being built concurrently in the same clone. The phase is therefore
split along the line that lets the first half merge regardless of Phase 5's
state:

**Change 1 — additive.** `infra/`, this spec, ADR-0020/0018/0019, the
deployment guide, nav, and the CI job that gates the infra. **Nothing under
`src/` changes**, so there is no file this change and a Phase 5 change can both
touch.

**Change 2 — instrumentation.** The `opentelemetry-api` base dependency, the
`otel` extra, the tracer helper, spans on `Indexer.run` and `Retriever.search`,
the JSON log formatter, the image gaining `--extra otel`, and the compose
stack's first real trace. Rebased on `origin/main` immediately before it opens,
so Phase 5's landed work is integrated over rather than around.

The cost of the split, stated so it is not mistaken for an oversight: **after
change 1 the compose stack runs a collector and a Jaeger that receive nothing.**
The topology is real and the pipeline can be proved with a synthetic OTLP
payload (§6.2), but no groundkit span reaches it until change 2. Every document
that describes the stack says this in the place a reader would otherwise assume
otherwise.

## 4. Decisions

### 4.1 Settled before the code — three ADRs

- **[ADR-0020](../adr/ADR-0020-terraform-target-single-host-with-block-storage.md)
  — the Terraform target.** AWS, one EC2 instance, one attached EBS volume, no
  ingress rules at all, access by SSM port forwarding. The decisive argument is
  storage, not provider preference: SQLite's WAL shared-memory mapping is not
  reliable over NFS, which disqualifies every serverless-container-plus-managed-
  filesystem answer (Fargate+EFS and its equivalents) before compute is even
  considered.
- **[ADR-0021](../adr/ADR-0021-container-exposure-and-filesystem-hardening.md) —
  what containerisation does to two Phase 4 guarantees.** The `127.0.0.1` bind
  cannot hold inside a container, so it is re-established at the publish
  boundary in all three surfaces; and the read-only claim splits, with
  `/data/corpus` becoming a genuinely read-only mount while `/data/index`
  cannot be one while WAL is in use.
- **[ADR-0022](../adr/ADR-0022-observability-dependency-shape-and-span-attribute-allowlist.md)
  — the observability shape.** `opentelemetry-api` in base so instrumentation
  needs no guarded imports, SDK and exporter in an `otel` extra, standard
  `OTEL_*` environment configuration rather than groundkit config keys, and an
  allowlist — not a denylist — for span attributes.

### 4.2 Probes have no health endpoint to point at, and inventing one is a `src/` decision this change does not take

`ADR-0014` decision 2 check 2 is a parity test: the set of `(path, method)`
pairs the app registers must equal the set the registry declares, with FastAPI's
own documentation routes excluded by an explicit `DOC_PATHS` frozenset. That
test is the mechanism by which "this surface exposes exactly the four declared
read-only operations" is a property rather than a claim.

A `/healthz` route is a fifth pair. Adding it means widening that exclusion set,
which is a change to a security-relevant invariant, in `src/`, in a phase whose
first change deliberately touches no `src/` file.

**So the probes target `GET /v1/collections`,** and the limitation is stated
rather than papered over:

- It proves the process is accepting and serving HTTP, and that the index
  directory is listable. That is a real check — it is not a static handler.
- It does **not** prove any collection is usable. `handle_list_collections`
  returns `[]` for a missing or empty index directory rather than failing, so a
  pod with an unmounted volume reports ready.
- Using a real operation as a **liveness** probe has a specific failure mode
  worth naming: if the index volume hangs, the probe hangs, the kubelet
  restarts the pod, and the restart does not fix a hung volume. The manifests
  therefore give liveness a long period and a high failure threshold, so it
  reacts to a wedged process rather than to storage weather.

A dedicated health endpoint — its route excluded from the registry parity set
deliberately and with a test asserting the exclusion is exactly one path — is a
named obligation on the phase that next touches `service/api.py`, not an
omission.

### 4.3 The image must bind `0.0.0.0`, and the flag that admits it stays

ADR-0021 decision 1. Recorded here as well because it is the single most
surprising line in `infra/docker/Dockerfile`: the default command contains
`--allow-remote-access`, and it is correct that it does. What used to be a
refusal is now a warning, and the manifests are what make the guarantee true.

### 4.4 Pinned image tags are a claim, and this repo does not make unverified claims quietly

Every third-party image in the compose stack is pinned by tag. None of those
tags was pulled while this change was written — no container runtime was
available (§6). A CI step therefore resolves each pinned reference with
`docker manifest inspect`, which is cheap, needs no layer download, and fails
the build on a tag that does not exist. That is the difference between a pin
and a guess.

## 5. What is built

```
infra/
  README.md                      verification-status table; read first
  docker/Dockerfile              multi-stage, non-root, read-only-root-ready
  compose/docker-compose.yml     service + ollama + otel-collector + jaeger
  compose/otel-collector.yaml    OTLP in, OTLP to Jaeger + debug out
  compose/.env.example           names only; no values, ever
  k8s/                           namespace, pvc, deployment, service, ingest Job
  terraform/aws-ec2/             ADR-0020's module
```

Change 2 adds, under `src/`: `groundkit/telemetry.py` (tracer accessor and the
typed-keyword attribute helper ADR-0022 decision 3 requires), a
`logging.Formatter` subclass, spans in `indexer.py` and `retrieval/search.py`,
and `--extra otel` in the image build.

## 6. Verification — what was executed, and what was not

SPEC.md §1.4 requires each IaC path be "verified to work, with the verification
date recorded." SPEC.md §2 forbids publishing a number, or a date, that was not
generated by a run. Where those meet, the honest answer is a status table with
some rows unfilled, and this is it.

### 6.1 Environment as found

| Tool | State on the machine this change was written on |
|---|---|
| `docker` CLI | present |
| Docker **daemon** | **not running** — `npipe:////./pipe/dockerDesktopLinuxEngine` unreachable |
| `docker compose` CLI | present (validates files; cannot build or run without the daemon) |
| `kubectl` | present, **no cluster context configured** |
| `terraform` | **not installed** |

Nothing was built there, no image was pulled, no container ran, no manifest was
applied to a cluster, and no Terraform plan was produced.

That is what the CI job in §6.2 is for, and it is not a formality: a runner has
a daemon and registry access, so on this change's first CI run the image built,
ran as uid 10001 under a read-only root, and all six pinned third-party tags
resolved — verifications this machine could not perform at all.
`infra/README.md` carries the status table, marks which rows were earned on a
runner, and is the file to update when a path is actually exercised.

### 6.2 The verification each path is owed

Recorded as the procedure, not as a result. A date goes into `infra/README.md`
only next to a row someone has actually run.

1. **Image.** `docker build -f infra/docker/Dockerfile -t groundkit:local .`,
   then `docker run --rm groundkit:local --version` and a check that the
   container's user is `10001` and its root filesystem is read-only.
2. **Compose.** `docker compose -f infra/compose/docker-compose.yml up -d`;
   ingest the sample corpus through the one-shot run; `curl` the loopback
   publish for `GET /v1/collections` and a `POST /v1/search`; confirm the
   published ports are loopback-only (`docker compose ps` and a bind check from
   another host on the LAN, which must fail).
3. **The trace pipeline, in two parts.** Before change 2, the collector→Jaeger
   leg is provable on its own by POSTing a synthetic OTLP/HTTP trace payload to
   the collector's `:4318/v1/traces` and finding it in the Jaeger UI. After
   change 2, the whole leg is provable the way SPEC.md §9 asks: issue a real
   search and find the `retrieve` span in Jaeger, with the attribute allowlist
   holding — no query text anywhere in the span.
4. **Kubernetes.** `kubectl kustomize infra/k8s` renders offline and is gated in
   CI. A genuine verification is `kubectl apply -k`, the ingest Job completing,
   the deployment reaching ready, and `kubectl port-forward` serving a search.
   That needs a cluster.
5. **Terraform.** `terraform fmt -check` and `terraform init -backend=false &&
   terraform validate` are gated in CI and prove the module is well-formed and
   its provider schema is satisfied. They do **not** prove it applies. A real
   verification is an apply into a throwaway account, an SSM port-forward
   session, and a search over the tunnel.

CI gates 1, 4 (render only) and 5 on every pull request, so those three cannot
rot silently. 2 and 3 need a running daemon and are the operator's to run.

## 7. Risks and open questions

**R1 — The compose stack's Ollama leg is the one part that is not air-gap
friendly out of the box.** Four images must be pulled once, and the embedding
model must be pulled once into the `ollama-models` volume. After that the stack
makes no outbound connection: groundkit's only outbound call is to the
configured embedding endpoint, and the collector exports to a sibling container.
The mitigation is documentation plus a pre-seeded volume, and it is the honest
shape of "air-gap friendly" for a stack that includes a model server — the
model has to arrive somehow.

**R2 — A pinned tag that has aged out.** The pins in the compose file were
chosen for confidence that they exist, not for recency, and no runtime confirmed
them. The `docker manifest inspect` CI step turns "probably exists" into a gate,
but a tag that exists is not a tag that is current. Refresh them at the first
real verification and record the date beside them.

**R3 — Instrumentation lands in files Phase 5 may also be editing.**
`retrieval/search.py` is the likeliest overlap. Mitigated by ordering: change 2
rebases on `origin/main` and integrates over whatever Phase 5 has landed, rather
than opening in parallel and merging blind. If the overlap turns out to be
sharp, the retrieve span lands and the ingest span waits — spans are independent
of one another in a way most changes are not.

**R4 — The core-subset coverage gate and a new `src/` module.**
`groundkit/telemetry.py` sits at the package root, so no glob in
`[tool.groundkit.coverage].core_subset` catches it, and `runtime.py`'s
precedent shows root modules are added to that list *explicitly* when they
belong there. Telemetry does not decide whether an answer is correct or current,
which is the test `runtime.py`'s entry was argued on — so the expectation is
that it stays out of the core subset and is covered by the whole-package gate
like `service/`. Decided in change 2 with the reasoning written into
`pyproject.toml` beside the existing entries, either way.

**Q1 — Does a health endpoint land, and at what cost to the parity test?**
§4.2 defers it. The question is not whether a probe target would be nicer — it
would — but whether widening `DOC_PATHS`-style exclusions is the right shape,
or whether the health route should live on a second ASGI app on a second port
that the registry parity test never sees. The second option keeps the
four-operations invariant literally intact and costs a port. Not decided here;
owed by whichever change next touches `service/api.py`.

**Q2 — Does anything ever make `/data/index` a read-only mount?** ADR-0021
decision 2 says not while WAL is in use, because `SQLiteMetadataStore.open`
writes on every open. A read-only open path in `index/metadata.py` — immutable
URI, `PRAGMA query_only`, no `CREATE TABLE IF NOT EXISTS`, no chmod — would make
a genuinely read-only serving deployment possible and would strengthen
ADR-0014's read-only claim from a Python-level scoping to a kernel-level one.
That is a real `src/` change with its own ADR and is not smuggled into a mount
flag.
