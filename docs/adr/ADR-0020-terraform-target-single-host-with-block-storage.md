# ADR-0020 — The Terraform module targets one AWS host with attached block storage, reachable only through SSM

- **Status:** Accepted (owner, 2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

SPEC.md §1.4 requires "real IaC: Dockerfile, compose, Kubernetes manifests, Terraform
module — each verified to work, with the verification date recorded," and §9's Phase 6
row narrows the last of those to "Terraform module for one concrete provider." Choosing
that provider is the decision this record exists for, and it is close to irreversible in
the way that matters: a module's resource graph, its IAM model, its storage primitive and
its access path are not portable between clouds, so a second provider is a second module
rather than a parameterisation of the first.

Three properties of what is being deployed constrain the choice far more than provider
preference does.

**The index is a filesystem, not a service.** ADR-0002 makes SQLite the durable truth,
one store per collection at `<index_dir>/<collection>.sqlite3` in WAL mode, with a
LanceDB `.lance` directory as its sibling. Both are local file formats. There is no
managed-database target to point at, and no object store either: LanceDB reads its
tables through a filesystem path and SQLite's WAL needs a real one.

**SQLite WAL rules out network filesystems.** WAL mode coordinates readers and writers
through a shared-memory file (the `-shm` sidecar) that SQLite maps into memory. That
mapping is not reliable over NFS, which is the storage model behind the obvious
serverless-container answer on every cloud (EFS on AWS, Filestore on GCP, Azure Files).
A container platform whose only persistent storage is an NFS mount is therefore not a
deployment target for this application — it is a corruption hazard wearing a deployment
target's clothes. The storage primitive has to be a block device.

**The application is single-node by specification.** SPEC.md §4 puts distributed
indexing out of scope for v1 and KNOWN_LIMITATIONS.md repeats it: one node, file-based
index. Nothing about this workload benefits from a platform whose value is horizontal
scale, and a platform that assumes horizontal scale will fight the RWO storage this
application requires.

One further property decides the *network* half rather than the compute half.
**Phase 4 ships no authentication of any kind.** ADR-0014 decision 1 makes every
operation read-only, which satisfies SPEC.md §7's shared-secret clause vacuously, and
decision 7 therefore makes the loopback bind the service's only access control. A
`search` response carries document text and absolute source paths. Any cloud module that
opens an inbound path to the service port publishes the corpus to whoever can reach it,
and does so in a deployment where — unlike a laptop — "whoever can reach it" is not
bounded by the operator's own machine.

## Decision

### 0. Amendments after review

Two claims in the first draft of this record were wrong in the same way — each described
a capability the resources did not actually provide — and are corrected in place above
rather than left for a reader to trip over:

- `create_ssm_vpc_endpoints` was described as making a NAT-free subnet work. It covers
  the SSM control channel only; bootstrap still needs egress. See decision 3.
- The instance role carried `AmazonSSMManagedInstanceCore` alone while this module's own
  README documented a private ECR image, which cannot be pulled without a login. The
  module now detects an ECR reference in `container_image`, attaches
  `AmazonEC2ContainerRegistryReadOnly` only in that case, and authenticates in
  bootstrap. See decision 6.

### 1. The provider is AWS, and the shape is one EC2 instance with an attached EBS volume

`infra/terraform/aws-ec2/` provisions a single instance, one encrypted `gp3` EBS volume
attached to it, an instance profile, and a security group. The instance runs the
container image built from `infra/docker/Dockerfile`.

EBS is a block device, so SQLite WAL behaves exactly as it does on a laptop. EC2 is the
one AWS compute primitive that offers a block device without also offering a scheduler
that wants to move the workload. The pairing is chosen for the storage model first and
the compute model second, which is the correct order for this application.

The module takes a `vpc_id` and a `subnet_id` rather than creating a network. Networking
is the part of a cloud footprint an organisation already owns and already has opinions
about; a module that creates its own VPC is a module that cannot be used inside one.

### 2. There is no inbound path to the service port, and that is a resource-level fact rather than a documented recommendation

The security group the module creates has **no ingress rules at all**. Not a
CIDR-restricted rule, not a rule referencing another group: none. On the host, the
container publishes to `127.0.0.1:8765`, mirroring what `infra/compose/` does and what
ADR-0014 decision 7 requires of the process.

Access is by AWS Systems Manager Session Manager port forwarding, which connects to
`localhost` *on the instance* and tunnels over the SSM agent's outbound channel. The
operator reaches the service at `127.0.0.1` on their own machine, authenticated by IAM,
authorised by IAM policy, and logged in CloudTrail. `terraform output
ssm_port_forward_command` prints the exact invocation.

This is the decision that makes the module honest. A module with a
`allowed_cidr_blocks` variable would hand the operator a supported, documented way to
publish an unauthenticated corpus, and the variable's existence would read as
endorsement. There is no such variable, and adding one is a change to this record.

### 3. Egress is restricted to HTTPS, and a private subnet is supported without NAT

The security group's only egress rule is TCP 443. That is what SSM, container registry
pulls and package installation need; nothing else in the deployed system makes an
outbound connection, because the embedding endpoint is an input (decision 5) rather than
a default reaching out to the internet.

`create_ssm_vpc_endpoints` (default `false`) creates the three interface endpoints —
`ssm`, `ssmmessages`, `ec2messages` — that let the **agent** reach Systems Manager
without a NAT gateway or an internet gateway. It defaults off because an account that
already has them, or already has NAT, would otherwise pay twice.

**It does not make a subnet with no egress work, and this record originally implied it
did.** Bootstrap installs docker from Amazon Linux's CDN and pulls a container image;
neither is reachable through those three endpoints, so `set -e` ends provisioning before
the service is installed and the operator is left with an instance they can open a
session to and no service running on it. The intended shape is therefore a private
subnet **with NAT**. A genuinely egress-free deployment is possible and is not built:
it needs an AMI with docker and the image already baked in, or the `ecr.api`/`ecr.dkr`
interface endpoints plus the `s3` gateway endpoint for layer storage *and* a
VPC-reachable package mirror. Recorded as a design rather than discovered as a failed
`terraform apply`.

IMDSv2 is required (`http_tokens = "required"`). The instance metadata endpoint is
link-local, which `utils/url_safety.py` refuses for outbound provider endpoints even when
private endpoints are permitted; requiring IMDSv2 closes the same class of hazard one
layer down, where a request forged through a proxy cannot mint a token.

### 4. Both volumes are encrypted, and the data volume outlives the instance

The EBS data volume and the instance's root volume are both encrypted with the account's
default KMS key. The data volume is a separate `aws_ebs_volume` with its own lifecycle
rather than an extra `ebs_block_device` on the instance, so replacing the instance — an
AMI change, an instance-type change — detaches and reattaches the index rather than
destroying it.

SPEC.md §7 is explicit that the SQLite store is content-bearing data, not just an index,
and that deletion behaviour, permissions, backup scope and retention are product
decisions owed before any non-local deployment. This decision settles exactly one of
them: the volume is not destroyed with the instance. **Backup and retention are not
settled here and the module creates no snapshot schedule** — see Consequences.

### 5. The module deploys groundkit, and does not deploy Ollama

`embedding_base_url` is an input with no default. Left unset, the instance runs the
BM25-only path, which is what the default install does everywhere else and what SPEC.md
§10 makes the definition of working end-to-end with zero cloud credentials.

Ollama is deliberately absent. Running it on the same instance would mean either CPU
inference on a machine sized for a retrieval service — slow enough to be misleading about
what the deployment can do — or a GPU instance class an order of magnitude more
expensive, which is a cost decision the module has no business making silently. The
compose stack, which runs on a machine the operator already owns, is where the local
Ollama topology lives.

Whatever this variable is set to, the **`groundkit-ingest` helper the bootstrap writes
takes the same setting.** That is not symmetry for its own sake: an ingest without
`--dense` against a deployment serving `--dense` produces a collection with no vectors
and no embedding-identity manifest, so every dense and hybrid request is refused
(ADR-0008) — on a deployment explicitly configured for them, with an error naming an
index inconsistency rather than the ingest three steps upstream. The two are rendered
from one input for that reason.

### 6. The role grants ECR pull only when the image is an ECR reference

`container_image` is matched against the ECR host pattern. When it matches,
`AmazonEC2ContainerRegistryReadOnly` is attached and bootstrap runs
`aws ecr get-login-password | docker login` against that registry and region; when it
does not, neither happens and the role keeps SSM alone.

Detected rather than asked for, because the alternative is a flag whose omission breaks
the module's own documented example silently — an unauthenticated pull of a private ECR
image fails under `set -e` before the systemd unit is written, which presents as an
instance that came up healthy and serves nothing. Detection also keeps the grant narrow:
a deployment pulling from a public registry never receives ECR permissions at all.

The registry and region come from the image reference rather than from
`data.aws_region`, so a cross-region pull is expressed by writing the cross-region
reference and needs no second variable.

## Alternatives considered

- **AWS ECS Fargate with an EFS volume.** The obvious serverless-container answer, and
  the one this decision rejects most firmly. EFS is NFS; SQLite's WAL shared-memory
  mapping is unreliable over it, so the deployment would either fail loudly on a lock or
  — far worse — corrupt quietly. LanceDB over NFS is no better characterised. This is not
  a preference, it is a compatibility fact about the storage engine ADR-0002 chose.
- **AWS ECS/EC2 with a bind-mounted host volume.** Recovers the block device, but adds a
  scheduler that can place the task on an instance whose volume holds a different
  collection, and pays a control plane for a service that will never have a second
  replica. The failure it introduces — a task rescheduled onto the wrong host — is silent
  and looks like an empty index.
- **A Terraform module targeting Kubernetes.** Rejected as duplication: `infra/k8s/`
  already expresses this deployment as Kubernetes objects, and wrapping them in
  `kubernetes_manifest` resources would produce a second copy that drifts from the first
  while proving nothing the manifests do not already prove. SPEC.md §9 lists k8s
  manifests and a Terraform module as *separate* deliverables, which is what makes the
  duplication a poor reading of the requirement rather than an efficient one.
- **A Terraform module targeting the local Docker daemon
  (`kreuzwerker/docker`).** The best fit for the local-first posture and the only option
  verifiable on a laptop with no cloud account, which is genuinely attractive. Rejected
  because it duplicates `infra/compose/` exactly — same daemon, same images, same
  volumes — and because "one concrete provider" in a phase whose sibling deliverables are
  already local reads as the cloud path being the gap. A module nobody would use in
  preference to `docker compose up` is a module written to satisfy a checklist.
- **Fly.io.** A very good architectural fit: single instance, real block-device volumes,
  private-by-default networking. Rejected on tooling risk rather than architecture — the
  Terraform provider is community-maintained and has not had the stability commitment
  that a first-party provider carries, and a deployment path this repo will verify once
  and then leave alone should not rest on one.
- **Hetzner Cloud, DigitalOcean.** First-party providers, cheap, block storage, and both
  would work. Rejected in favour of AWS because SSM Session Manager has no equivalent on
  either: reaching the service would mean SSH, which means a key to manage, a bastion or
  an inbound rule — that is, it would mean reopening decision 2. The access model, not
  the compute, is what AWS wins on here.
- **GCP Compute Engine with a persistent disk plus IAP TCP forwarding.** The closest
  genuine competitor; IAP forwarding is a real analogue of SSM port forwarding and
  persistent disks are block devices. Decided on familiarity and on the portfolio's
  existing AWS-shaped conventions rather than on any technical deficiency, and recorded
  that way rather than dressed up as a technical verdict it is not.

## Consequences

- Deploying costs money and requires an AWS account. Nothing in the local-first posture
  changes: `uv run grk serve`, `docker compose up`, and the k8s manifests all remain
  credential-free, and this module is opt-in infrastructure that no test, gate or default
  path touches.
- **Reaching the service requires the SSM plugin and IAM permissions.** There is no
  fallback URL, and that is the point. An operator without `ssm:StartSession` cannot
  reach the corpus, which is the same statement as "the corpus is not published."
- **Backups are not configured.** No DLM policy, no snapshot schedule, no lifecycle rule.
  The volume survives instance replacement and nothing else; a deleted volume is a
  deleted corpus. SPEC.md §7 names backup scope as a product decision owed before a
  non-local deployment, and this record settles durability-across-instance-replacement
  only. Anyone running this for real owes the rest of that decision, and the module's
  README says so where an operator will read it.
- **Single instance means single point of failure and downtime on every change.** An AMI
  refresh or instance-type change replaces the instance and the service is down until it
  returns. That is inherent to the single-node architecture SPEC.md §4 chose, not a
  shortcoming of the module.
- A second cloud is now a second module rather than a variable. That was true of any
  choice here; it is recorded so the next reader does not attempt to generalise this one.
- The module is written against a provider version range rather than a pin
  (`>= 5.40.0, < 7.0.0`). A consumer with a lockfile pins it; the module deliberately does
  not, because a module that pins a patch version fights every consumer that has one.

## References

- SPEC.md **§1.4** (real IaC, each verified with the verification date recorded), §4
  (distributed indexing out of scope), §7 (SQLite is content-bearing; deletion, backup
  and retention are product decisions owed before a non-local deployment), §9 (Phase 6
  row: "Terraform module for one concrete provider"), §10 (zero cloud credentials for the
  local definition of done).
- [ADR-0002](ADR-0002-index-persistence.md) — SQLite as durable truth, WAL mode, one
  store per collection; the storage model that makes a block device non-negotiable.
- [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md) —
  decision 1 (no mutating operations, so no authentication ships) and decision 7 (the
  loopback bind is the only access control), which together force decision 2 here.
- [ADR-0021](ADR-0021-container-exposure-and-filesystem-hardening.md) — how the same
  loopback guarantee is preserved when the process runs in a container, which is what the
  instance's `127.0.0.1` publish relies on.
- `infra/terraform/aws-ec2/README.md` — inputs, outputs, the access procedure, and the
  verification status of this path.
