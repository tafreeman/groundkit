# `aws-ec2` — groundkit on one instance with an attached block volume

The Terraform half of SPEC.md §9's Phase 6 row. The provider choice, its
alternatives and the reasoning are in
[ADR-0020](../../../docs/adr/ADR-0020-terraform-target-single-host-with-block-storage.md);
this file is the operating manual.

## What it creates

One EC2 instance running the groundkit container, one encrypted EBS data volume
with its own lifecycle, an instance profile carrying
`AmazonSSMManagedInstanceCore` (plus `AmazonEC2ContainerRegistryReadOnly` **only
when `container_image` is a private ECR reference**), and a security group with
**no ingress rules at all**. Optionally, three SSM interface endpoints.

It does **not** create a VPC, a subnet, a NAT gateway, a DNS record, a load
balancer, a backup schedule, or an Ollama.

## Network prerequisite, before you plan

The instance needs **outbound HTTPS during bootstrap**: it installs docker from
Amazon Linux's CDN and pulls the container image. A private subnet therefore
wants a NAT gateway.

`create_ssm_vpc_endpoints` is **not** a substitute. It creates the
`ssm`/`ssmmessages`/`ec2messages` endpoints, which carry the Session Manager
control channel and nothing else; neither the package repository nor a container
registry is reachable through them. Set it on an otherwise egress-free subnet
and you get an instance you can open a session to, running no service, because
`set -e` ended bootstrap at `dnf install`. A genuinely egress-free deployment
needs an AMI with docker and the image baked in, or `ecr.api` + `ecr.dkr`
interface endpoints plus the `s3` gateway endpoint *and* a VPC-reachable package
mirror — none of which this module builds.

## Private ECR images work, and that is not automatic

The `container_image` reference is matched against the private-ECR host pattern
(`<account>.dkr.ecr.<region>.amazonaws.com/…`). When it matches, the role gains
`AmazonEC2ContainerRegistryReadOnly` and bootstrap runs `aws ecr
get-login-password | docker login` against that registry and region. When it
does not — `ghcr.io`, Docker Hub, `public.ecr.aws`, all of which pull
anonymously — neither happens and the role keeps SSM alone.

Detected rather than flagged because the failure it prevents is silent: an
unauthenticated pull of a private image fails under `set -e` before the systemd
unit is written, so the instance comes up healthy and serves nothing.

## The access model, before anything else

Phase 4 ships **no authentication of any kind**
([ADR-0014](../../../docs/adr/ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md)
decision 1), and a `search` response carries document text and absolute source
paths. So this module gives the service no reachable address:

- the container publishes to `127.0.0.1` on the instance;
- the security group has no ingress rules, and there is **no variable that
  would create one** — adding one is a change to ADR-0020, not a configuration
  choice;
- there is no key pair and no SSH.

The way in is a Session Manager port-forwarding session, which connects to
`localhost` *on the instance* over the SSM agent's outbound channel:

```bash
terraform output -raw ssm_port_forward_command   # prints the exact invocation
# then, on your own machine:
curl http://127.0.0.1:8765/v1/collections
```

That requires the [Session Manager
plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
and `ssm:StartSession` on the instance. An operator without it cannot reach the
corpus, which is the same statement as "the corpus is not published."

## Usage

```hcl
module "groundkit" {
  source = "github.com/tafreeman/groundkit//infra/terraform/aws-ec2"

  name      = "groundkit"
  vpc_id    = "vpc-0123456789abcdef0"
  subnet_id = "subnet-0123456789abcdef0"

  # Required, no default: no groundkit image is published to any registry yet.
  # Build it from infra/docker/Dockerfile and push it somewhere this account
  # can pull from.
  container_image = "123456789012.dkr.ecr.eu-west-1.amazonaws.com/groundkit:0.1.0"

  # SSM control channel only — read "Network prerequisite" above before
  # assuming this replaces a NAT gateway. It does not.
  create_ssm_vpc_endpoints = true
}
```

Then, over an SSM session, put documents in `/srv/groundkit/corpus` and run:

```bash
sudo groundkit-ingest          # written by the bootstrap script
sudo systemctl restart groundkit
```

`groundkit-ingest` is rendered with the **same** embedding arguments as the
service unit, from the same `embedding_base_url` input. That matters: a
BM25-only ingest feeding a `--dense` service produces a collection with no
vectors and no embedding-identity manifest, so every dense and hybrid request is
refused ([ADR-0008](../../../docs/adr/ADR-0008-dense-search-requires-a-dense-collection.md))
with an error naming an index inconsistency rather than the ingest that caused
it.

The restart is not optional and the reason is a real property of the software: a
`Retriever` is a snapshot of the store as of `open()` and never refreshes, so a
process that was serving before the ingest returns zero results for everything
the ingest added. It fails closed on modified or deleted content and silently
short on new content — the silent half is why this line is here.

An **empty** index also serves happily: `list_collections` returns `[]` rather
than an error, so a freshly provisioned volume looks healthy and answers
nothing. Ingest before concluding the deployment works.

## Inputs

| Name | Type | Default | Notes |
|---|---|---|---|
| `name` | string | `groundkit` | Resource name prefix. |
| `vpc_id` | string | — | Required. |
| `subnet_id` | string | — | Required; a private subnet is the intended shape. |
| `container_image` | string | — | Required; nothing is published. |
| `instance_type` | string | `t3.small` | Memory is the dimension that matters — see below. |
| `root_volume_size_gb` | number | `20` | Encrypted. |
| `data_volume_size_gb` | number | `20` | Encrypted; grows with the corpus, not with a fixed overhead. |
| `data_volume_type` | string | `gp3` | Must stay a block type — ADR-0020 decision 1. |
| `collection` | string | `default` | Used by the `groundkit-ingest` helper. |
| `embedding_base_url` | string | `""` | Empty runs the BM25-only path. |
| `host_port` | number | `8765` | Loopback publish and SSM forward target. |
| `create_ssm_vpc_endpoints` | bool | `false` | SSM control channel only; not a NAT replacement. |
| `tags` | map(string) | `{}` | Merged onto everything. |

Sizing note: `BM25Index.from_store()` rebuilds postings **in memory at every
`Retriever.open()`** ([ADR-0002](../../../docs/adr/ADR-0002-index-persistence.md)),
so the working set scales with corpus size rather than with request rate. If the
instance runs out of memory, the corpus outgrew it — not the traffic.

## Outputs

`instance_id`, `instance_private_ip`, `data_volume_id`, `security_group_id`,
`iam_role_arn`, `ssm_port_forward_command`.

## What this module does not give you

- **Backups.** No snapshot schedule, no DLM policy. `prevent_destroy` on the
  data volume means it survives instance replacement and nothing more, and a
  deleted volume is a deleted corpus — SPEC.md §7 is explicit that the SQLite
  store holds document text, not a rebuildable index. Backup scope, retention
  and deletion behaviour are product decisions owed before any real deployment,
  and this module settles exactly one of them.
- **High availability.** One instance. An AMI refresh or an instance-type change
  replaces it and the service is down until it returns. That follows from
  SPEC.md §4's single-node architecture, not from the module.
- **Ollama.** ADR-0020 decision 5. Set `embedding_base_url` to something you
  already run.
- **Corpus delivery.** Getting documents to `/srv/groundkit/corpus` is yours to
  choose — `aws s3 sync` over the SSM session, a git clone, a baked layer.

## Verification status

| Check | Status |
|---|---|
| `terraform fmt -check -recursive` | **passed 2026-08-16** |
| `terraform validate`, AWS provider 6.60.0 | **passed 2026-08-16** |
| `terraform validate`, AWS provider 5.100.0 (the declared floor's major) | **passed 2026-08-15** |
| `bootstrap.sh.tftpl` renders for both the ECR/dense and public/BM25 cases; both pass `bash -n` | **passed 2026-08-16** |
| Rendered systemd unit emits single-backslash continuations | **passed 2026-08-15** |
| Rendered `groundkit-ingest` carries the service's embedding arguments | **passed 2026-08-16** |
| ECR detection matches private ECR (incl. GovCloud) and not ghcr/Docker Hub/`public.ecr.aws` | **passed 2026-08-16** |
| `terraform plan` against a real account | **not yet run** |
| `terraform apply`, SSM session, a search over the tunnel | **not yet run** |

`validate` proves the configuration is well-formed and satisfies the provider's
schema. It does **not** prove the module applies: it makes no API calls, so an
IAM permission that is missing, an instance type unavailable in the region, or
an AMI filter that matches nothing are all invisible to it. The last two rows
are what would close that, and neither has been run — see
`docs/specs/phase-6-iac-observability.md` §6 for the environment that made the
difference.
