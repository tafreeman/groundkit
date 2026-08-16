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
  EOT
  type        = string
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance type. Memory is the dimension that matters: BM25 postings are
    rebuilt in memory at every Retriever.open() (ADR-0002), so the working set
    scales with corpus size rather than with request rate.
  EOT
  type        = string
  default     = "t3.small"
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
  EOT
  type        = string
  default     = "gp3"
}

variable "collection" {
  description = "Collection name the service serves and the ingest writes to."
  type        = string
  default     = "default"
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
  EOT
  type        = string
  default     = ""
}

variable "host_port" {
  description = <<-EOT
    Port the container publishes on the instance's LOOPBACK interface, and the
    port an SSM port-forwarding session targets. It is never reachable from the
    network: the security group has no ingress rules (ADR-0020 decision 2).
  EOT
  type        = number
  default     = 8765
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

variable "tags" {
  description = "Extra tags merged onto every resource this module creates."
  type        = map(string)
  default     = {}
}
