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
}

data "aws_subnet" "this" {
  id = var.subnet_id
}

# Amazon Linux 2023: current, has the SSM agent preinstalled, and ships a
# maintained docker package. Resolved as a data source rather than pinned so a
# rebuild picks up patches; the cost is that `terraform plan` can show an
# instance replacement after an upstream AMI release, which is the correct
# trade for a single-instance deployment whose replacement is cheap.
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
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "this" {
  name_prefix = "${var.name}-"
  role        = aws_iam_role.this.name
  tags        = local.tags
}

# -- Network: no way in -----------------------------------------------------

resource "aws_security_group" "instance" {
  name_prefix = "${var.name}-"
  description = "groundkit: no ingress; egress HTTPS only (ADR-0020 decision 2/3)."
  vpc_id      = var.vpc_id
  tags        = local.tags

  lifecycle {
    create_before_destroy = true
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
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
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
  })
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
data "aws_vpc_endpoint_service" "ssm" {
  for_each = var.create_ssm_vpc_endpoints ? toset(["ssm", "ssmmessages", "ec2messages"]) : toset([])

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
