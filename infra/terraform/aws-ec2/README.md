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

The subnet must supply that egress itself, because the instance takes **no
public IPv4**. `associate_public_ip_address` is pinned `false` rather than left
to inherit the subnet's `MapPublicIpOnLaunch` — otherwise a public subnet gave
the host a public address while this module's own outputs said "no public
address, no ingress rule, no SSH". So a public subnet is not the shape here:
with no public IP there is no route out through an internet gateway.

`create_ssm_vpc_endpoints` is **not** a substitute. It creates the endpoints in
`ssm_vpc_endpoint_services` — by default `ssm` and `ssmmessages` — which carry
the Session Manager control channel and nothing else; neither the package
repository nor a container registry is reachable through them. Set it on an
otherwise egress-free subnet and you get an instance you can open a session to,
running no service, because `set -e` ended bootstrap at `dnf install`. A
genuinely egress-free deployment needs an AMI with docker and the image baked
in, or `ecr.api` + `ecr.dkr` interface endpoints plus the `s3` gateway endpoint
*and* a VPC-reachable package mirror — none of which this module builds.

`ec2messages` is **not** in the default set. It is the legacy channel — the SSM
agent on the AL2023 image this module pins uses `ssm` and `ssmmessages` — and it
is not offered in every region. `data.aws_vpc_endpoint_service` errors on a
service its region does not offer, and that error fails the whole `plan` rather
than the one endpoint, so including it by default let an optional convenience
break a deployment that never needed it. Add it back through the variable if you
run an older agent, in a region you have confirmed offers it.

## The dense path needs an egress rule, and it is derived from the URL

The security group's standing egress rule is TCP 443. Setting
`embedding_base_url` adds a second rule for that endpoint's port when it is not
443 — which is the usual case, because Ollama listens on **11434**. Without it
the module planned, applied and booted a deployment configured for dense
retrieval whose every embed call timed out at the security group.

Host and port are parsed from the URL rather than taken as separate inputs. Two
cases cannot be derived, and both fail at `plan` naming the reason rather than at
request time with a timeout:

| `embedding_base_url` | What you need |
|---|---|
| empty | nothing — BM25-only, no rule |
| `https://embed.internal` (port 443) | nothing — the standing rule covers it |
| `http://10.0.0.5:11434` | nothing — an IPv4 literal is its own `/32` |
| `http://ollama.internal:11434` | `embedding_egress_cidr` — a DNS name resolves on the instance, not here |
| any IPv6 endpoint, **any port** | refused at `plan`; this module writes `cidr_ipv4` rules only |
| a query string, `#fragment`, or `user:pw@` | refused at `plan` — `utils/url_safety.validate_endpoint_shape` refuses all three, so accepting them here would provision an instance whose service fails at startup |

The IPv6 refusal does not depend on the port, and that is load-bearing rather than
pedantic: the standing HTTPS rule is `cidr_ipv4 = 0.0.0.0/0`, which does not
carry IPv6 either. "Port 443 is already covered" is only true of IPv4.

## x86_64 only, and `instance_type` is checked against it

The AMI filter selects `al2023-ami-2023.*-x86_64` and the container image is
built by a plain `docker build` on an amd64 runner, so a Graviton
`instance_type` is refused by this module rather than by EC2 at launch. Adding
arm64 means a multi-architecture image build first; it is not a different value
for that variable. `g4dn` and `g6` are x86 despite the leading `g` and are
accepted.

## Private ECR images work, and that is not automatic

The `container_image` reference is matched against the private-ECR host pattern
(`<account>.dkr.ecr.<region>.amazonaws.com/…`, and `.amazonaws.com.cn` in the
China partition). When it matches, the role gains
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

`bootstrap.sh.tftpl` starts the container with `--host=0.0.0.0
--allow-remote-access`, because inside the container `127.0.0.1` would be
unreachable even from the instance's own SSM agent. That means the service's
`Host`-header check
([ADR-0024](../../../docs/adr/ADR-0024-host-header-validation-on-both-transports.md))
is **off** on this instance — `derive_host_allow_list` disables it the moment
the bind is routable, and `0.0.0.0` always is. Nothing in this module relies
on it. The access model above — no ingress rule, no key pair, SSM port
forwarding only — is the entire boundary, exactly as it was before ADR-0024,
and it has to be: a security group with no ingress rules is what makes the
instance unreachable in the first place, and `Host` validation only matters
once a request already arrives.

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
service unit, from the same `embedding_base_url` input, by the same Terraform
interpolation — the helper is no longer assembled by a `sed` pass over
placeholders, where an `&` in the URL corrupted the helper while the unit kept
the real value. That matters: a
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
| `subnet_id` | string | — | Required; must be in `vpc_id` (checked at `plan`); a private subnet is the intended shape. |
| `container_image` | string | — | Required; nothing is published. |
| `instance_type` | string | `t3.small` | **x86_64 only** — validated. Memory is the dimension that matters, see below. |
| `root_volume_size_gb` | number | `20` | Encrypted. |
| `data_volume_size_gb` | number | `20` | Encrypted; grows with the corpus, not with a fixed overhead. |
| `data_volume_type` | string | `gp3` | `gp3` or `gp2` only — io1/io2 need `iops`, st1/sc1 have a 125 GiB minimum. |
| `collection` | string | `default` | Used by the `groundkit-ingest` helper. |
| `embedding_base_url` | string | `""` | Empty runs the BM25-only path; otherwise the egress rule is derived from it. |
| `embedding_egress_cidr` | string | `""` | Only for a DNS-name endpoint off 443 — see above. |
| `host_port` | number | `8765` | Loopback publish and SSM forward target; a whole number in 1-65535. |
| `create_ssm_vpc_endpoints` | bool | `false` | SSM control channel only; not a NAT replacement. |
| `ssm_vpc_endpoint_services` | list(string) | `["ssm","ssmmessages"]` | Which endpoints that creates. `ec2messages` is opt-in — see above. |
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
| `terraform validate`, AWS provider 5.100.0 (the declared floor's major) | **passed 2026-08-16** |
| `bootstrap.sh.tftpl` renders for both the ECR/dense and public/BM25 cases; both pass `bash -n` | **passed 2026-08-16** |
| Rendered systemd unit emits single-backslash continuations | **passed 2026-08-15** |
| Rendered unit carries `RequiresMountsFor=/srv/groundkit`; bootstrap refuses an unmounted `/srv/groundkit` before creating anything under it | **passed 2026-08-16** |
| Rendered `groundkit-ingest` carries the service's embedding arguments, and the unit carries the same endpoint | **passed 2026-08-16** |
| A `%` in the endpoint URL renders `%%` in the unit and stays `%` in the helper | **passed 2026-08-16** |
| Storage prep renders as a retrying oneshot and resolves the Xen `/dev/xvdf` name; the script it writes passes `bash -n` | **passed 2026-08-16** |
| ECR detection matches private ECR in the commercial, GovCloud and China partitions, and not ghcr/Docker Hub/`public.ecr.aws` | **passed 2026-08-16** |
| Egress derivation classifies sixteen `embedding_base_url` forms, including an IPv6 host flagged at every port | **passed 2026-08-16** |
| The input validations reject what they claim to — Graviton types, shell metacharacters, bad ports, query/fragment/userinfo | **passed 2026-08-16** |
| The security-group **preconditions** actually reject | **passed 2026-08-16** — `terraform test` with `mock_provider`, 5 runs, no credentials and no API call. Preconditions evaluate at plan time and `terraform test` runs a plan, so mocks reach them; `tests/security_group_preconditions.tftest.hcl` |
| `terraform plan` against a real account | **passed 2026-08-16** — `us-east-1`, a real personal sandbox account; 9 resources to add, 0 to change/destroy, real AMI resolved, ECR detection matched a real private-ECR reference |
| `terraform apply`, SSM session, a search over the tunnel | **passed 2026-08-16** — instance booted, bootstrap pulled the image from the private ECR repo and mounted the data volume, `groundkit-ingest` indexed a planted document, and a real `POST /v1/search` over an SSM port-forward tunnel returned the correct citation-bearing result. Full scope (NAT gateway prerequisite, what was and wasn't exercised) recorded in `infra/README.md`'s verification status section — read it before assuming more than this run tested. |

`validate` proves the configuration is well-formed and satisfies the provider's
schema. It does **not** prove the module applies: it makes no API calls, so an
IAM permission that is missing, an instance type unavailable in the region, or
an AMI filter that matches nothing are all invisible to it. The last three rows
are what closed that, and **all three have now been run** — the preconditions
under `terraform test` with mocked providers, and the `plan` and `apply` against
a real account on 2026-08-16. See `docs/specs/phase-6-iac-observability.md` §6
for the environment, and `infra/README.md` for the apply's full scope.

Everything above those last three rows was executed locally on 2026-08-16 with
`terraform console`, which resolves locals, variable validations and
`templatefile()` without configuring a provider — so it needs no credential.
**None of it is gated.** These are point-in-time local runs, not a check that
would contradict them if they stopped being true. A CI gate for exactly these
derivations exists and is parked on `chore/infra-ci-checks-parked` rather than
shipped here: it re-tests Terraform's own validation engine more than it tests
this module, and it is not where the module's real risk lies.

That risk was the last three rows, and they are now closed — but by two
different instruments, and conflating them would overstate what either proves.

`validate` and `console` make no API call, and a `precondition` is reachable by
neither. **`terraform test` with `mock_provider` does reach them**, because
preconditions evaluate at plan time and `terraform test` runs a plan: that is
what closed the precondition row, with no credential and no API call. Recorded
carefully, because a mocked suite was previously declined for this module and
**that reasoning still stands** — mocks cannot catch a missing IAM permission,
an instance type unavailable in the region, or an AMI filter matching nothing.
The decline answered "can mocks substitute for a real apply?" (no). The suite
answers the narrower "can mocks close the never-executed-precondition row?"
(yes). Both remain true.

The three failure modes mocks cannot see are exactly what the `plan` and
`apply` rows closed on 2026-08-16, against a real account. One finding from
writing the suite is worth keeping: **precondition 2 is unreachable through the
variable path.** `variables.tf`'s URL validation regex is a strict superset of
the prefix `local.embed_scheme` derives from, so any value clearing validation
necessarily yields a non-empty scheme. It is defensive-only, and the suite
documents that rather than pretending to exercise it.