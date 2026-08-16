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

Every claim corrected here was wrong in one way: it described a capability the resources
did not actually provide. They are corrected in place above rather than left for a reader
to trip over. Note the shape of the class — each one applied, booted and passed a probe,
and failed later at a distance from its cause.

Round one:

- `create_ssm_vpc_endpoints` was described as making a NAT-free subnet work. It covers
  the SSM control channel only; bootstrap still needs egress. See decision 3.
- The instance role carried `AmazonSSMManagedInstanceCore` alone while this module's own
  README documented a private ECR image, which cannot be pulled without a login. The
  module now detects an ECR reference in `container_image`, attaches
  `AmazonEC2ContainerRegistryReadOnly` only in that case, and authenticates in
  bootstrap. See decision 6.

Round two, on the fixes themselves — two of the three are defects the round-one fixes
introduced or made reachable, which is the argument for reviewing a fix as a change
rather than as a correction:

- **Egress was 443-only while the module rendered `--dense` into both the service unit
  and the ingest helper.** Round one made the helper agree with the service; both then
  agreed on an endpoint the security group blocked. See decision 3.
- **Managed-policy ARNs were hardcoded to the `aws` partition, and the ECR host matcher
  stopped at `.amazonaws.com`.** Both are inert in the commercial partition and both fail
  in GovCloud or China — the ARN at role creation, the matcher by silently skipping the
  login it exists to perform. The ECR matcher is round one's own code. See decision 6.
- **The service unit ordered itself after the data mount without requiring it.** With
  `nofail` in fstab — which decision 2 requires, since there is no SSH to fall back on —
  a reboot where the volume does not attach started the service anyway and let docker
  create the bind-mount paths on the root disk. `RequiresMountsFor=/srv/groundkit` now
  makes the dependency real, and bootstrap verifies the mount before creating anything
  under it. The instance still boots and SSM still works; only the service fails, with
  the mount named in `systemctl status`.

Round three, both findings against round two's own fixes. The pattern has now repeated
three times and is worth stating as a rule rather than an observation: **a fix is a
change and inherits the full review a change gets.** Round one's ECR matcher carried the
partition defect; round two's egress fix carried the IPv6 defect.

- **The IPv6 rejection was derived from the port.** It rode on the same "does this need a
  rule?" flag as the CIDR check, and that flag is false at 443 — so an HTTPS IPv6 literal
  skipped the rejection entirely and applied onto a security group with no IPv6 egress at
  all. Now independent of port. See decision 3.
- **`instance_type` was an unrestricted string over a single-architecture deployment.**
  The AMI filter selects `x86_64` and the container image is built by a plain
  `docker build` on an amd64 runner, so a Graviton type produced a module that cannot
  apply. Refused at the input rather than accommodated by selecting an arm64 AMI, because
  the *image* is the binding constraint: an arm64 host would boot, pass its SSM check and
  then fail to pull, moving a loud EC2 launch rejection into a silent bootstrap failure.
  See decision 7.
- **The ingest helper was written by a `sed` pass over placeholders, and `sed` gives its
  replacement text its own syntax.** An unescaped `&` there means "the whole match", so an
  `embedding_base_url` carrying a query string wrote the literal token
  `EMBED_PLACEHOLDER` into the helper while the systemd unit, built by ordinary
  interpolation, kept the real URL. That is round one's defect through a different door —
  the helper and the service disagreeing about the endpoint, ending in an ADR-0008
  refusal that names an index inconsistency rather than its cause. The placeholders and
  the `sed` pass are gone: the helper is interpolated like everything else, which removes
  the escaping question instead of answering it. See decision 8.
- **`create_ssm_vpc_endpoints` hardcoded `ec2messages` into a strict data source.** A
  service name a region does not offer is an error, and a failing data source fails the
  whole plan — so an optional endpoint the AL2023 agent does not even use could take the
  module down. Now `ssm_vpc_endpoint_services`, defaulting to the two that agent uses.
  See decision 3.

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

### 3. Egress is HTTPS plus the configured embedding endpoint, and a private subnet wants NAT

The standing egress rule is TCP 443 to `0.0.0.0/0`. That is what SSM, container registry
pulls and package installation need.

It is **not** all the deployed system needs, which this record originally got wrong by
reasoning from the wrong half of decision 5. The embedding endpoint being an input rather
than a default means the module does not reach out to the internet on its own — it does
not mean nothing is reached. When `embedding_base_url` is set, the instance dials it on
every ingest and every dense or hybrid request, and Ollama's default port is 11434. A
443-only group let that configuration plan, apply and boot, and then time out at the
security group on the first embed call, with the service's own logs showing an
unreachable endpoint rather than a rule that forbade it.

The rule is therefore derived from the URL — host and port parsed out, one additional
egress rule when the port is not 443, scoped to that endpoint rather than widening the
standing rule. Derived rather than configured because a separate `embedding_port` input
is a second thing that has to agree with the first. Two cases cannot be derived and both
fail at `plan` with the reason named: a DNS-name host (nothing here resolves it, so
`embedding_egress_cidr` supplies the CIDR) and an IPv6 endpoint (this module writes
`cidr_ipv4` rules only). Failing at plan is the point — the alternative is a timeout on a
deployment that applied cleanly.

**The IPv6 rejection is independent of the port, and briefly was not.** The first version
of it rode on the same "does this need a rule?" flag as the CIDR check, which is false at
port 443 because the standing rule covers that port. That reasoning is only true for
IPv4. An `https://[2001:db8::10]` endpoint therefore skipped the rejection, created no
rule, and applied cleanly onto a security group whose every rule — the standing one
included — is `cidr_ipv4`. An address family is not a function of a port, and deriving
one from the other reintroduced precisely the silent timeout this decision exists to
remove.

This is also the one outbound destination that carries query text, so it gets its own
rule and its own description rather than being absorbed into a broader one.

`create_ssm_vpc_endpoints` (default `false`) creates the interface endpoints that let the
**agent** reach Systems Manager without a NAT gateway or an internet gateway. It defaults
off because an account that already has them, or already has NAT, would otherwise pay
twice.

Which endpoints is `ssm_vpc_endpoint_services`, defaulting to `ssm` and `ssmmessages`,
and it was a hardcoded three including `ec2messages`. Two facts make that the wrong
default. `ec2messages` is the legacy channel — the SSM agent shipped on the AL2023 image
this module pins uses `ssm` and `ssmmessages` — and it is not offered in every region.
The second fact is the one with teeth: `data.aws_vpc_endpoint_service` errors on a
service the region does not offer, and a failing data source fails **the entire plan**,
not the one endpoint. So an optional convenience could take the whole module down over an
endpoint the deployment had no use for. An operator running an older agent adds it back
through the variable, in a region they have confirmed offers it.

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

**The matcher and the policy ARNs are partition-aware, and were not.** A private ECR
registry in `cn-north-1` or `cn-northwest-1` is `<account>.dkr.ecr.<region>.amazonaws.com.cn`,
which a matcher ending at `.amazonaws.com` does not match — so in the one partition where
an operator is least able to shrug it off, the module skipped both the policy attachment
and the `docker login`, and the pull failed exactly the way this decision exists to
prevent. The AWS-managed policy ARNs had the partition hardcoded as `aws` for the same
reason: nothing in the commercial partition can tell. Both now derive from
`data.aws_partition.current`, which is the same reasoning that already sent the VPC
endpoint service names through a data source rather than string interpolation.

What this does **not** claim: that the module has been applied in GovCloud or China. It
has not been applied anywhere — see `infra/README.md`'s status table. These two fixes
remove two known-wrong strings; they are not a statement that the remaining ones are
right, and the EC2 service principal in the assume-role policy is the next thing a reader
should check before trying it.

### 7. The deployment is x86_64, and `instance_type` is validated rather than accommodated

The AMI data source filters `al2023-ami-2023.*-x86_64`. `instance_type` was an
unrestricted string, so an ARM family — `t4g.small`, say — produced a module that could
not apply: EC2 rejects a launch whose instance architecture and image architecture
disagree.

The interesting part is which of the two knobs to turn. Selecting the AMI architecture
*from* the instance type is the more capable answer and is the wrong one here, because the
AMI is not the binding constraint. `infra/docker/Dockerfile` is built by CI with a plain
`docker build` on an amd64 runner, so the image has a single manifest. An arm64 instance
with a correctly-matched arm64 AMI would launch, boot, register with SSM, and then fail at
`docker pull` inside bootstrap under `set -e` — which is this module's signature failure:
an instance that looks healthy and serves nothing. Deriving the AMI would have converted a
loud, immediate, well-worded EC2 error into a silent one three layers down.

So the input is validated instead, and the error message names the two reasons and says
what would actually be needed. Supporting arm64 is a multi-architecture image build plus
this filter, in that order — not a different value for this variable.

The validation matches Graviton families (`a1`, and any family carrying a `g` in the
suffix after the generation digit) and deliberately does not match `g4dn`/`g6`, where the
`g` is the family letter and the instance is x86. A family it fails to recognise falls
through to EC2's own rejection, which is the pre-existing behaviour and is loud.

### 8. Operator-supplied strings are constrained at the input, and the generated files interpolate rather than substitute

Three inputs are written into files on the instance: `container_image` and
`embedding_base_url` into both the ingest helper and the systemd unit, and `collection`
into the helper. Two things now hold about that, and neither did.

**Interpolation, not substitution.** The helper used to be written as a heredoc full of
`IMAGE_PLACEHOLDER`-style tokens and then rewritten by `sed`. That is a second escaping
context nobody asks for: `&` in a `sed` replacement means the matched text, `|` was the
delimiter, and a backslash escapes. Terraform substitutes these values when it renders
the template, long before the instance's shell sees anything, so the placeholders bought
nothing and cost an injection surface. The quoted heredoc stays — it is what keeps `$@`
literal when the file is written — but the values arrive already interpolated.

**A character class per input, not a quoting convention.** The values are single-quoted
in the generated script, and `collection`, `container_image` and `embedding_base_url` are
validated against classes that exclude the single quote, `$`, backticks, backslashes and
whitespace. Quoting alone would be a convention that the next edit to this template can
silently drop; the validation is what makes it a property. `collection` deliberately
mirrors `index/metadata.py`'s `_COLLECTION_NAME_PATTERN` and its `.`/`..` rejection, so a
name the application itself would refuse cannot reach an instance and fail there instead.

`&` remains legal in `embedding_base_url` — it is ordinary in a query string, and with
`sed` gone it is ordinary here too. The rule is about shell and template metacharacters,
not about punctuation in general.

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
