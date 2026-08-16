# groundkit on one EC2 instance with an attached EBS volume (ADR-0020).
#
# The shape in one paragraph: one instance, one encrypted block volume with its
# own lifecycle, a security group with NO ingress rules at all, and an instance
# profile whose only permission is what Session Manager needs. The container
# publishes to 127.0.0.1 on the instance; the way in is an SSM port-forwarding
# session, authenticated by IAM and recorded in CloudTrail. There is no URL.
#
# That is not defence in depth around an authenticated service — Phase 4 ships
# no authentication of any kind (ADR-0014 decision 1), and a `search` response
# carries document text and absolute source paths. The absence of an inbound
# path is the entire access control, which is why it is a property of the
# resources here rather than a recommendation in the README.

locals {
  tags = merge(
    {
      "app.kubernetes.io/name" = var.name
      ManagedBy                = "terraform"
      Module                   = "groundkit/aws-ec2"
    },
    var.tags,
  )

  # Attached as /dev/sdf; on a Nitro instance the kernel presents it as an NVMe
  # device instead, so the bootstrap script resolves it by volume id under
  # /dev/disk/by-id rather than trusting this name.
  data_device_name = "/dev/sdf"

  # Whether `container_image` names a private ECR registry, and if so which one.
  # Detected rather than asked for: the module's own README documents an ECR
  # reference as the usual case, and an unauthenticated `docker pull` of one
  # fails under `set -e` before the service is ever written — so a flag the
  # operator had to remember would be a flag whose omission silently breaks the
  # documented path. Deriving it also keeps the IAM grant below narrow: the ECR
  # policy attaches only when there is genuinely an ECR image to pull.
  #
  # The `.cn` suffix is not decoration. A private ECR registry in cn-north-1 or
  # cn-northwest-1 is `<account>.dkr.ecr.<region>.amazonaws.com.cn`, and a
  # matcher that stops at `.amazonaws.com` skips both the policy attachment and
  # the `docker login` for it — the same silent failure this detection exists to
  # prevent, in the partition where it is hardest to debug. Non-capturing so the
  # group indices below stay put.
  ecr_match    = regexall("^([0-9]{12}\\.dkr\\.ecr\\.([a-z0-9-]+)\\.amazonaws\\.com(?:\\.cn)?)/", var.container_image)
  ecr_registry = length(local.ecr_match) > 0 ? local.ecr_match[0][0] : ""
  ecr_region   = length(local.ecr_match) > 0 ? local.ecr_match[0][1] : ""

  # The embedding endpoint's egress, derived from the URL rather than asked for
  # as a separate host/port pair: two inputs that have to agree are one input
  # and a latent bug.
  #
  # The standing egress rule below is TCP 443 and nothing else, which is right
  # for the SSM channel, the package CDN and a registry — and wrong for the
  # endpoint this module's own variable documentation gives as the usual case.
  # Ollama listens on 11434, so a deployment configured for dense retrieval had
  # `groundkit-ingest` and every dense and hybrid request time out at the
  # security group, with nothing in the service's own logs to say why: the
  # module was configured for the dense path and silently could not reach it.
  embed_match  = var.embedding_base_url == "" ? [] : regexall("^(https?)://(\\[[^]]+\\]|[^/:?#]+)(?::([0-9]+))?", var.embedding_base_url)
  embed_scheme = length(local.embed_match) > 0 ? local.embed_match[0][0] : ""
  embed_host   = length(local.embed_match) > 0 ? local.embed_match[0][1] : ""
  embed_port = length(local.embed_match) == 0 ? 0 : (
    local.embed_match[0][2] != null
    ? tonumber(local.embed_match[0][2])
    : (local.embed_scheme == "https" ? 443 : 80)
  )

  # An IPv4 literal is its own /32. A DNS name resolves on the instance at
  # request time and there is nothing here to resolve it from, so that case has
  # to be given — see the precondition on the security group.
  embed_cidr = (
    var.embedding_egress_cidr != ""
    ? var.embedding_egress_cidr
    : (can(cidrnetmask("${local.embed_host}/32")) ? "${local.embed_host}/32" : "")
  )

  # A bracketed host is an IPv6 literal, and this module has NO IPv6 egress at
  # all: every rule it writes is `cidr_ipv4`, the standing HTTPS rule included.
  # So the address family is rejected on its own terms, independently of port.
  #
  # It was not, and the bug that hid in the difference is worth keeping: the
  # rejection used to ride on `embed_needs_egress`, which is false at port 443
  # because the standing rule covers it. That reasoning holds only for IPv4. An
  # `https://[2001:db8::10]` endpoint therefore skipped the rejection, created
  # no rule, and applied cleanly onto a security group with no IPv6 egress
  # whatsoever — every embedding request timing out with nothing refused.
  # Deriving a rejection from a port was the mistake; an address family is not
  # a function of a port.
  embed_is_ipv6 = startswith(local.embed_host, "[")

  # No second rule when the endpoint is already covered: the HTTPS rule is
  # 0.0.0.0/0, so an IPv4 endpoint on 443 needs nothing further, and a duplicate
  # would be one more rule to read in the console for no reachability.
  embed_needs_egress = local.embed_scheme != "" && !local.embed_is_ipv6 && local.embed_port != 443
}

# The partition, so the AWS-managed policy ARNs below are not commercial-only.
# This module already resolves VPC endpoint service names through a data source
# for exactly this reason; a hardcoded `arn:aws:` is the same class of mistake
# one layer up, and it fails at role creation in GovCloud and China.
data "aws_partition" "current" {}

data "aws_subnet" "this" {
  id = var.subnet_id
}

# Amazon Linux 2023: current, has the SSM agent preinstalled, and ships a
# maintained docker package. Resolved as a data source rather than pinned so a
# rebuild picks up patches; the cost is that `terraform plan` can show an
# instance replacement after an upstream AMI release, which is the correct
# trade for a single-instance deployment whose replacement is cheap.
#
# x86_64 is hardcoded here and `instance_type` is validated against it, because
# this filter is only half the constraint: the container image is built by a
# plain `docker build` on an amd64 runner and has one manifest. Deriving the
# architecture from the instance type instead would launch an arm64 host that
# boots, passes its SSM check, and then fails to pull the image at bootstrap --
# trading EC2's loud launch rejection for a silent one. The variable's
# validation is where that is refused; see its description.
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# -- IAM: Session Manager and nothing else ----------------------------------

data "aws_iam_policy_document" "assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "this" {
  name_prefix        = "${var.name}-"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.tags
}

# The AWS-managed policy, deliberately, rather than a hand-written subset. It is
# the documented minimum for Session Manager and it tracks the service; a
# hand-copied subset would silently stop working when SSM adds an action, and
# the failure would look like a broken instance rather than a stale policy.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.this.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Pull-only, and attached only when `container_image` is an ECR reference. The
# managed read-only policy rather than a hand-written one, for the same reason
# as the SSM attachment above: it is the documented minimum and it tracks the
# service. It grants no push and no repository administration.
resource "aws_iam_role_policy_attachment" "ecr_pull" {
  count = local.ecr_registry == "" ? 0 : 1

  role       = aws_iam_role.this.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "this" {
  name_prefix = "${var.name}-"
  role        = aws_iam_role.this.name
  tags        = local.tags
}

# -- Network: no way in -----------------------------------------------------

resource "aws_security_group" "instance" {
  name_prefix = "${var.name}-"
  description = "groundkit: no ingress; egress is HTTPS plus the configured embedding endpoint (ADR-0020 decision 2/3)."
  vpc_id      = var.vpc_id
  tags        = local.tags

  lifecycle {
    create_before_destroy = true

    # `terraform validate` does not evaluate a precondition; `plan` does, which
    # is early enough. What it replaces is a request-time timeout on a
    # deployment that validated, planned and applied without a complaint.
    # The subnet must be IN the VPC this group is created in. `variables.tf`
    # has always said so and nothing enforced it: the group is created in
    # `vpc_id`, the instance launches in `subnet_id`, and EC2 rejects the
    # combination at `RunInstances` — after the role, the profile, the volume
    # and this group already exist. The data source knows the answer, so the
    # documented constraint becomes a plan-time one.
    precondition {
      condition     = data.aws_subnet.this.vpc_id == var.vpc_id
      error_message = "subnet_id belongs to VPC ${data.aws_subnet.this.vpc_id}, not to vpc_id ${var.vpc_id}. EC2 would reject the instance launch after the other resources were created."
    }

    precondition {
      condition     = var.embedding_base_url == "" || local.embed_scheme != ""
      error_message = "embedding_base_url must be an http:// or https:// URL. Got: ${var.embedding_base_url}"
    }

    # Independent of port, and that independence is the whole point: every rule
    # this module writes is `cidr_ipv4`, so an IPv6 endpoint is unreachable
    # whatever port it is on. Gating this on `embed_needs_egress` made the
    # rejection an accident of the port number and let `https://[2001:db8::10]`
    # through.
    precondition {
      condition = !local.embed_is_ipv6
      error_message = join(" ", [
        "embedding_base_url names the IPv6 endpoint ${local.embed_host}, which this",
        "module cannot reach: every egress rule it writes is cidr_ipv4, including",
        "the standing HTTPS rule. Use an IPv4 or DNS-named endpoint, or add IPv6",
        "egress to this module first.",
      ])
    }

    precondition {
      condition = !local.embed_needs_egress || local.embed_cidr != ""
      error_message = join(" ", [
        "embedding_base_url resolves to ${local.embed_host}:${local.embed_port},",
        "which the HTTPS egress rule does not cover, and its host is not an IPv4",
        "literal this module can turn into a /32.",
        "Set embedding_egress_cidr to the IPv4 CIDR the endpoint is reachable at.",
      ])
    }
  }
}

# There is deliberately NO aws_vpc_security_group_ingress_rule in this module,
# and no variable that would create one. A CIDR-restricted ingress rule would be
# a supported, documented way to publish an unauthenticated corpus, and the
# variable's existence would read as endorsement. Adding one is a change to
# ADR-0020, not a configuration choice.

resource "aws_vpc_security_group_egress_rule" "https" {
  security_group_id = aws_security_group.instance.id
  description       = "SSM control channel, registry pulls, package updates."
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
  tags              = local.tags
}

# The dense path's egress, and the only rule here that is not 443. Created only
# when `embedding_base_url` names a port the rule above does not already cover,
# and scoped to that one endpoint rather than widening the standing rule: this
# is the single outbound destination that carries query text, so it earns its
# own rule with its own description rather than hiding inside a broader one.
#
# Absent it, the module accepted `embedding_base_url`, rendered `--dense` into
# both the service unit and the ingest helper, and then let every embed call
# time out at the security group.
resource "aws_vpc_security_group_egress_rule" "embedding" {
  count = local.embed_needs_egress ? 1 : 0

  security_group_id = aws_security_group.instance.id
  description       = "Embedding endpoint for the dense path (${local.embed_host}:${local.embed_port})."
  ip_protocol       = "tcp"
  from_port         = local.embed_port
  to_port           = local.embed_port
  cidr_ipv4         = local.embed_cidr
  tags              = local.tags
}

# -- Storage: a block device with its own lifecycle -------------------------

resource "aws_ebs_volume" "data" {
  availability_zone = data.aws_subnet.this.availability_zone
  size              = var.data_volume_size_gb
  type              = var.data_volume_type
  encrypted         = true
  tags              = merge(local.tags, { Name = "${var.name}-data" })

  # A separate volume rather than an extra ebs_block_device on the instance, so
  # replacing the instance — an AMI refresh, an instance-type change — detaches
  # and reattaches the index instead of destroying it (ADR-0020 decision 4).
  #
  # What this does NOT provide: backups. No snapshot schedule, no DLM policy. A
  # deleted volume is a deleted corpus, and SPEC.md §7 names backup scope as a
  # product decision owed before any non-local deployment.
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_volume_attachment" "data" {
  device_name = local.data_device_name
  volume_id   = aws_ebs_volume.data.id
  instance_id = aws_instance.this.id
}

# -- Compute ----------------------------------------------------------------

resource "aws_instance" "this" {
  ami           = data.aws_ami.al2023.id
  instance_type = var.instance_type
  subnet_id     = var.subnet_id

  # Explicitly false, because unset does not mean "no". Unset inherits the
  # subnet's `MapPublicIpOnLaunch`, so dropping this module into a public
  # subnet handed the instance a public IPv4 — while `outputs.tf` told the
  # operator "no public address, no ingress rule, no SSH". The security group
  # and the loopback publish still keep the service unreachable, so this is not
  # an exposure of the corpus; it is an unnecessary public surface on the host,
  # a billed IPv4, and a claim the resources did not back.
  #
  # It also makes the documented shape a requirement rather than a preference:
  # with no public IP there is no route out through an internet gateway, so the
  # subnet must provide egress itself. That is the private-subnet-with-NAT shape
  # this module already asks for, and bootstrap fails at `dnf install` without
  # it either way.
  associate_public_ip_address = false

  vpc_security_group_ids = [aws_security_group.instance.id]
  iam_instance_profile   = aws_iam_instance_profile.this.name
  ebs_optimized          = true
  monitoring             = true
  tags                   = merge(local.tags, { Name = var.name })

  # No key_name, deliberately: there is no SSH ingress for a key to be used
  # with, and an unused key pair is a credential to manage for nothing.

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_size_gb
    encrypted   = true
    tags        = merge(local.tags, { Name = "${var.name}-root" })
  }

  # IMDSv2 required. The metadata endpoint is link-local, which
  # utils/url_safety.py refuses for outbound provider endpoints even when
  # private endpoints are permitted; requiring a session token closes the same
  # class of hazard one layer down, where a request forged through a proxy
  # cannot mint one.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "disabled"
  }

  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/bootstrap.sh.tftpl", {
    container_image    = var.container_image
    host_port          = var.host_port
    collection         = var.collection
    embedding_base_url = var.embedding_base_url
    data_volume_id     = aws_ebs_volume.data.id
    data_device_name   = local.data_device_name
    ecr_registry       = local.ecr_registry
    ecr_region         = local.ecr_region
  })

  # The instance profile has to exist before the instance boots, or the
  # bootstrap's `aws ecr get-login-password` runs with no credentials. Terraform
  # infers the profile dependency from the attribute reference above but not the
  # policy attachments hanging off the role, so they are named explicitly.
  depends_on = [
    aws_iam_role_policy_attachment.ssm,
    aws_iam_role_policy_attachment.ecr_pull,
  ]
}

# -- Optional: reach SSM from a subnet with no NAT --------------------------

resource "aws_security_group" "endpoints" {
  count = var.create_ssm_vpc_endpoints ? 1 : 0

  name_prefix = "${var.name}-vpce-"
  description = "HTTPS from the groundkit instance to the SSM interface endpoints."
  vpc_id      = var.vpc_id
  tags        = local.tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_vpc_security_group_ingress_rule" "endpoints_https" {
  count = var.create_ssm_vpc_endpoints ? 1 : 0

  security_group_id            = aws_security_group.endpoints[0].id
  description                  = "HTTPS from the instance only."
  ip_protocol                  = "tcp"
  from_port                    = 443
  to_port                      = 443
  referenced_security_group_id = aws_security_group.instance.id
  tags                         = local.tags
}

# The one ingress rule in this module, and it is on the ENDPOINT security group
# rather than the instance's: it admits the instance to the AWS API, not the
# world to the service. Source is the instance's security group, not a CIDR.

# Resolved through the service data source rather than interpolated as
# "com.amazonaws.<region>.<service>". Two reasons: the region would have to come
# from data.aws_region, whose attribute for it was renamed across the provider
# major this module's version range spans, and a hand-built service name is a
# string this module would get wrong in exactly the partitions (GovCloud, China)
# where nobody would be around to notice.
#
# The set is a VARIABLE and no longer the hardcoded three, because this data
# source is strict: a service name the region does not offer is an error, and it
# fails the whole plan rather than that one endpoint. `ec2messages` was in the
# hardcoded set and is the one that provokes it — it is the legacy channel, not
# required by the SSM agent AL2023 ships (3.3+ uses ssm and ssmmessages), and
# not offered everywhere. So an optional convenience could take the module down
# in a region that has no use for the endpoint it was failing on.
#
# Default is therefore the two the pinned AMI's agent actually uses. An operator
# on an older agent adds "ec2messages" back through the variable, in a region
# they have already confirmed offers it.
data "aws_vpc_endpoint_service" "ssm" {
  for_each = var.create_ssm_vpc_endpoints ? toset(var.ssm_vpc_endpoint_services) : toset([])

  service = each.key
}

resource "aws_vpc_endpoint" "ssm" {
  for_each = data.aws_vpc_endpoint_service.ssm

  vpc_id              = var.vpc_id
  service_name        = each.value.service_name
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [var.subnet_id]
  security_group_ids  = [aws_security_group.endpoints[0].id]
  private_dns_enabled = true
  tags                = merge(local.tags, { Name = "${var.name}-${each.key}" })
}
