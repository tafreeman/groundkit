variable "name" {
  description = "Name prefix for every resource this module creates."
  type        = string
  default     = "groundkit"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{0,30}$", var.name))
    error_message = "name must be 1-31 lowercase alphanumerics or hyphens and start with an alphanumeric."
  }
}

variable "vpc_id" {
  description = <<-EOT
    VPC the instance and its security group live in. Required: this module does
    not create a network, because networking is the part of a cloud footprint an
    organisation already owns, and a module that creates its own VPC cannot be
    used inside one (ADR-0020 decision 1).
  EOT
  type        = string
}

variable "subnet_id" {
  description = <<-EOT
    Subnet for the instance. Must be in var.vpc_id. A private subnet with a NAT
    gateway is the intended shape: the instance needs outbound HTTPS during
    bootstrap for the docker package and the container image, and the SSM agent
    needs to reach Systems Manager or there is no way in at all (ADR-0020
    decision 2 opens no ingress).

    With no NAT, set create_ssm_vpc_endpoints for the SSM control channel — but
    read that variable's description first, because it does not make bootstrap
    work on its own.
  EOT
  type        = string
}

variable "container_image" {
  description = <<-EOT
    Fully-qualified container image reference for the groundkit service, built
    from infra/docker/Dockerfile. Required with no default: no groundkit image
    is published to any registry, so any default here would be a reference that
    cannot be pulled.

    Restricted to the OCI reference character set. This value is written into a
    generated shell script and a systemd unit, so the restriction is what makes
    the quoting there sufficient rather than hopeful.
  EOT
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9][A-Za-z0-9._:/@-]*$", var.container_image))
    error_message = "container_image must be an OCI reference: alphanumerics and . _ - : / @ only, starting with an alphanumeric."
  }
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance type. Memory is the dimension that matters: BM25 postings are
    rebuilt in memory at every Retriever.open() (ADR-0002), so the working set
    scales with corpus size rather than with request rate.

    **x86_64 only.** Two things in this deployment are single-architecture and
    neither is a variable: the AMI data source filters `al2023-ami-2023.*-x86_64`,
    and `infra/docker/Dockerfile` is built by CI with a plain `docker build` on an
    amd64 runner, so the published image has one manifest. A Graviton type is
    rejected here rather than at EC2, and arm64 support is a multi-architecture
    image build, not a change to this input.
  EOT
  type        = string
  default     = "t3.small"

  validation {
    # Graviton families: `a1`, and any family whose suffix begins with `g` after
    # the generation digit (t4g, m6gd, c7gn, im4gn, is4gen, x2gd, hpc7g, r8g).
    # Deliberately not matched: g4dn/g6, where the `g` is the family letter and
    # the instance is x86.
    condition     = !can(regex("^(a1|[a-z]+[0-9]+g[a-z]*)\\.", var.instance_type))
    error_message = "instance_type ${var.instance_type} looks like an arm64/Graviton family, and this module is x86_64 only: the AMI filter selects an x86_64 image and the container image is built for one architecture. Supporting arm64 means a multi-arch image build, not a different value here."
  }
}

variable "root_volume_size_gb" {
  description = "Size of the instance's encrypted root volume, in GiB."
  type        = number
  default     = 20
}

variable "data_volume_size_gb" {
  description = <<-EOT
    Size of the encrypted EBS data volume holding the index, in GiB. SPEC.md §7:
    the SQLite store holds document TEXT, not just an index, so this grows with
    the corpus rather than with a fixed metadata overhead.
  EOT
  type        = number
  default     = 20
}

variable "data_volume_type" {
  description = <<-EOT
    EBS volume type for the data volume. Must be a BLOCK device type — the
    reason this module exists in this shape is that SQLite's WAL shared-memory
    mapping is not reliable over a network filesystem (ADR-0020 decision 1).

    Restricted to `gp3` and `gp2`, which is narrower than "any block type" and
    narrower than this description used to promise. `io1`/`io2` require an
    `iops` argument this resource does not set, so volume creation is rejected;
    `st1`/`sc1` have a 125 GiB minimum that the 20 GiB default violates. Both
    fail at apply for a value the docs invited. Supporting them means exposing
    and validating the type-dependent `iops` and size, which is a larger
    surface than a single-instance retrieval index needs.
  EOT
  type        = string
  default     = "gp3"

  validation {
    condition     = contains(["gp3", "gp2"], var.data_volume_type)
    error_message = "data_volume_type must be gp3 or gp2. io1/io2 need an iops argument this module does not set, and st1/sc1 have a 125 GiB minimum the default size violates."
  }
}

variable "collection" {
  description = <<-EOT
    Collection name the service serves and the ingest writes to.

    Constrained to the same character class groundkit itself enforces
    (`index/metadata.py`'s `_COLLECTION_NAME_PATTERN`, plus its explicit `.`/`..`
    rejection), mirrored here so a name the application would refuse never
    reaches an instance — and so the value is safe to write into the generated
    ingest helper.
  EOT
  type        = string
  default     = "default"

  validation {
    condition = (
      can(regex("^[A-Za-z0-9._-]+$", var.collection))
      && var.collection != "."
      && var.collection != ".."
    )
    error_message = "collection must match ^[A-Za-z0-9._-]+$ and be neither '.' nor '..' — the same rule index/metadata.py enforces."
  }
}

variable "embedding_base_url" {
  description = <<-EOT
    Embedding endpoint for the dense retrieval path. Empty (the default) runs
    the BM25-only path, which is what a default install does everywhere else and
    what SPEC.md §10 makes the local definition of done. This module deploys no
    Ollama — ADR-0020 decision 5 records why.

    Note the outbound guard this crosses: OllamaEmbedder is the one provider
    exempted from the private-address refusal (ADR-0014 decision 10). An
    OpenAI-compatible endpoint on an RFC1918 address is refused at request time.

    The security group's egress follows this input. Its host and port are parsed
    out, and a port other than 443 gets its own egress rule — Ollama's default
    is 11434, so the documented dense deployment is exactly the case the
    standing HTTPS rule does not cover. A DNS-name host additionally needs
    `embedding_egress_cidr`, because nothing here can resolve it at plan time.
  EOT
  type        = string
  default     = ""

  # The trailing `([/?#]|$)` is the load-bearing part. Without it the match was
  # unanchored past the host, so `http://10.0.0.5:abc` matched through
  # `10.0.0.5` and stopped — the port-range rule below then saw no numeric port
  # and passed vacuously, the locals inferred the scheme default of 80 and built
  # an egress rule for it, and bootstrap handed groundkit the original `:abc`.
  # Applies cleanly, opens the wrong port, fails when the HTTP client parses the
  # endpoint. Requiring the authority to END means a colon must be followed by
  # digits or the whole thing is refused.
  validation {
    condition     = var.embedding_base_url == "" || can(regex("^https?://(\\[[^]]+\\]|[^/:?#]+)(:[0-9]+)?([/?#]|$)", var.embedding_base_url))
    error_message = "embedding_base_url must be empty or an http:// or https:// URL whose authority ends cleanly — a colon must be followed by a numeric port."
  }

  # A URL character set that excludes the quote, backtick, dollar, backslash and
  # whitespace. `&` is deliberately still allowed — it is legal in a query string
  # and is now harmless, because the ingest helper is written by interpolation
  # rather than by a `sed` replacement where `&` meant "the whole match".
  validation {
    condition     = var.embedding_base_url == "" || can(regex("^https?://[A-Za-z0-9._~:/?#\\[\\]@!&()*+,;=%-]+$", var.embedding_base_url))
    error_message = "embedding_base_url may contain only unreserved URL characters: quotes, backticks, $, backslash and whitespace are rejected because this value is written into a generated shell script and a systemd unit."
  }

  # The module must not accept an endpoint shape the APPLICATION refuses.
  # `utils/url_safety.validate_endpoint_shape` rejects a query string or
  # fragment outright — request paths are concatenated onto base_url, so one
  # there would attach to every request — and rejects userinfo, because a
  # credential belongs in `api_key_env`. A URL carrying either passes Terraform,
  # provisions an instance, and then fails when `grk serve --dense` constructs
  # the embedder, which is the familiar shape: applied cleanly, serving nothing.
  validation {
    condition     = var.embedding_base_url == "" || !can(regex("[?#]", var.embedding_base_url))
    error_message = "embedding_base_url must not carry a query string or fragment: utils/url_safety.validate_endpoint_shape rejects both, so the service would fail at startup on an instance that applied cleanly."
  }

  validation {
    condition     = var.embedding_base_url == "" || !can(regex("^https?://[^/]*@", var.embedding_base_url))
    error_message = "embedding_base_url must not carry userinfo: utils/url_safety.validate_endpoint_shape rejects it, and a credential belongs in api_key_env rather than in a URL."
  }

  # An explicit port becomes `from_port`/`to_port` on a real egress rule, and
  # the provider does not catch every bad value: port 0 produces a rule that
  # applies cleanly and permits nothing usable, so the module comes up with the
  # dense path configured and permanently unreachable. Above 65535 fails later,
  # in the provider, which is a worse place to learn it. `regexall` yields an
  # empty list when no port is given, and `alltrue([])` is true — so a URL
  # relying on the scheme default skips this check rather than failing it.
  validation {
    condition = var.embedding_base_url == "" || alltrue([
      for m in regexall("^https?://(?:\\[[^]]+\\]|[^/:?#]+):([0-9]+)", var.embedding_base_url) :
      tonumber(m[0]) >= 1 && tonumber(m[0]) <= 65535
    ])
    error_message = "embedding_base_url's explicit port must be between 1 and 65535."
  }
}

variable "embedding_egress_cidr" {
  description = <<-EOT
    IPv4 CIDR the embedding endpoint is reachable at, for the egress rule this
    module derives from `embedding_base_url`.

    Needed only when that endpoint's port is not 443 AND its host is a DNS name.
    An IPv4-literal host is its own /32 and needs nothing here; a host on 443 is
    already covered by the standing egress rule. Anything else fails at `plan`
    with the reason, rather than at request time with a timeout.

    IPv6 endpoints are not supported: this module writes `cidr_ipv4` rules only.
  EOT
  type        = string
  default     = ""

  validation {
    condition     = var.embedding_egress_cidr == "" || can(cidrnetmask(var.embedding_egress_cidr))
    error_message = "embedding_egress_cidr must be empty or an IPv4 CIDR such as 10.0.0.0/24."
  }
}

variable "host_port" {
  description = <<-EOT
    Port the container publishes on the instance's LOOPBACK interface, and the
    port an SSM port-forwarding session targets. It is never reachable from the
    network: the security group has no ingress rules (ADR-0020 decision 2).

    A whole number in 1-65535. `number` alone accepts `8765.5` and `70000`,
    which reach both docker's publish mapping and the `ssm_port_forward_command`
    output — the instance then applies successfully and the service never
    starts, or the command the module prints as "the way in" is unusable.
  EOT
  type        = number
  default     = 8765

  validation {
    condition = (
      floor(var.host_port) == var.host_port
      && var.host_port >= 1
      && var.host_port <= 65535
    )
    error_message = "host_port must be a whole number between 1 and 65535."
  }
}

variable "create_ssm_vpc_endpoints" {
  description = <<-EOT
    Create the ssm/ssmmessages/ec2messages interface endpoints, so the SSM agent
    reaches Systems Manager without traversing a NAT or internet gateway. Off by
    default so an account that already has them, or already has NAT, does not
    pay twice.

    This covers the SSM CONTROL CHANNEL ONLY. It does not make a subnet with no
    egress work: bootstrap runs `dnf install -y docker` against Amazon Linux's
    CDN and then pulls the container image, and neither is reachable through
    these three endpoints. Setting this on a subnet with no other egress leaves
    an instance you can open a session to and a service that was never
    installed. See the module README for what a genuinely egress-free
    deployment would take.
  EOT
  type        = bool
  default     = false
}

variable "ssm_vpc_endpoint_services" {
  description = <<-EOT
    Which SSM interface endpoints `create_ssm_vpc_endpoints` creates.

    The default is what the SSM agent on this module's pinned AL2023 AMI
    actually uses. `ec2messages` is deliberately NOT in it: it is the legacy
    channel — agent 3.3+ uses `ssm` and `ssmmessages` — and it is not offered in
    every region. That matters more than it sounds, because
    `data.aws_vpc_endpoint_service` errors on a service the region does not
    offer, and the error fails the entire plan rather than that one endpoint. An
    optional convenience could therefore take the whole module down over an
    endpoint the deployment had no use for.

    Add `"ec2messages"` if you run an older agent, in a region you have
    confirmed offers it.
  EOT
  type        = list(string)
  default     = ["ssm", "ssmmessages"]

  validation {
    condition = (
      length(var.ssm_vpc_endpoint_services) > 0
      && length(setsubtract(var.ssm_vpc_endpoint_services, ["ssm", "ssmmessages", "ec2messages"])) == 0
    )
    error_message = "ssm_vpc_endpoint_services must be a non-empty subset of [\"ssm\", \"ssmmessages\", \"ec2messages\"]."
  }
}

variable "tags" {
  description = "Extra tags merged onto every resource this module creates."
  type        = map(string)
  default     = {}
}
