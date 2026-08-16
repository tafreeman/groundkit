# terraform test suite: security-group precondition coverage
# =============================================================================
#
# Why this file exists (docs/specs/phase-6-iac-observability.md §6.2 item 5):
#
#   "Nothing offline reaches a `precondition` -- `validate` does not evaluate
#   one and `console` cannot -- so every precondition in the module is
#   unexecuted."
#
# Terraform 1.15.8 evaluates `precondition` blocks at PLAN time, and
# `mock_provider` (Terraform 1.7+) builds a plan with no AWS credentials and
# no API call. That closes the gap the quote above describes: every
# precondition on aws_security_group.instance (main.tf:206-239) can be
# reached and exercised offline.
#
# SCOPE BOUNDARY -- read this before extending the suite:
#
#   A `terraform test` suite for this module was previously evaluated and
#   DECLINED on the grounds that mock_provider cannot catch a missing IAM
#   permission, an unavailable instance type, or an AMI filter that matches
#   nothing. That reasoning stands, and this suite does not reverse it: none
#   of those failure classes are reachable through a mocked plan, and nothing
#   here claims otherwise. What changed is narrower -- "can mocks close the
#   never-executed-precondition row in §6.2 item 5?" -- and the answer is
#   yes. This file is that answer, not a substitute for a real `apply` into a
#   throwaway account, which §6.2 item 5 itself names as the only thing that
#   closes the module's actual exposure.
#
# PRECONDITION 2 -- REACHABILITY FINDING:
#
#   main.tf:211-214 has a precondition guarding that embedding_base_url is
#   either empty or an http(s) URL (`var.embedding_base_url == "" ||
#   local.embed_scheme != ""`). Investigated whether any input that passes
#   this module's own variable validation can still reach that precondition
#   with local.embed_scheme == "".
#
#   It cannot. variables.tf:177-180's validation already requires any
#   non-empty value to match
#     ^https?://(\[[^]]+\]|[^/:?#]+)(:[0-9]+)?([/?#]|$)
#   and local.embed_scheme (main.tf:59-60) comes from a regexall using
#     ^(https?)://(\[[^]]+\]|[^/:?#]+)(?::([0-9]+))?
#   The scheme/host/port portion of both patterns is character-for-character
#   identical (same alternation for the host, same optional-port group); the
#   variable-validation pattern only adds a mandatory `[/?#]`-or-end-of-string
#   requirement *after* that portion. A string that satisfies the stricter,
#   longer pattern necessarily satisfies its own prefix, so embed_scheme is
#   never "" for a non-empty embedding_base_url that made it past variable
#   validation. Combined with the `var.embedding_base_url == ""` left-hand
#   disjunct, precondition 2 is therefore ALWAYS true for any input that
#   reaches plan -- it is defensive-only, unreachable through the variable
#   path today, and would only start firing if variables.tf:177-180's regex
#   were loosened or removed later.
#
#   "not-a-url", the input that motivated this investigation, is rejected by
#   variable validation (variables.tf:177 and :186) and never reaches a plan
#   at all. run.embedding_base_url_invalid_url_rejected_at_variable_validation
#   below demonstrates that boundary instead of pretending to exercise
#   precondition 2. No run block in this file targets precondition 2 with
#   expect_failures against aws_security_group.instance for it, and none
#   should be added unless variables.tf's URL-shape validation changes.
#
# MOCKS -- why each one is here:
#
#   - aws_iam_policy_document: the AWS provider validates
#     aws_iam_role.assume_role_policy as JSON (main.tf:156). mock_provider's
#     synthetic default for an unconfigured computed string attribute is a
#     random placeholder, which is not valid JSON and fails that
#     provider-side validation before any precondition is ever reached -- so
#     this data source's mocked `json` output has to be a real assume-role
#     policy document.
#   - aws_partition: data.aws_partition.current.partition is interpolated
#     into two IAM policy ARNs (main.tf:166, :177); given real "aws" /
#     "amazonaws.com" values so those ARNs, and any diagnostics involving
#     them, stay readable.
#   - aws_subnet: data.aws_subnet.this.vpc_id feeds precondition 1 directly,
#     and .availability_zone feeds aws_ebs_volume.data (main.tf:283). This is
#     the data source overridden per-run (via override_data) to drive
#     precondition 1 -- proof that a mock reaches a data-source-dependent
#     precondition, not only a pure-locals one.
#   - aws_ami: aws_instance.this.ami reads data.aws_ami.al2023.id
#     (main.tf:329); given an explicit realistic id rather than depending on
#     mock_provider's random-string default for the AMI the instance
#     launches.
#   - aws_vpc_endpoint_service: only reached when create_ssm_vpc_endpoints is
#     true, which no run below sets. Mocked anyway so a future run that does
#     exercise that path does not silently inherit a random-string default in
#     place of a real AWS service name.

mock_provider "aws" {
  mock_data "aws_iam_policy_document" {
    defaults = {
      json = jsonencode({
        Version = "2012-10-17"
        Statement = [
          {
            Effect    = "Allow"
            Action    = "sts:AssumeRole"
            Principal = { Service = "ec2.amazonaws.com" }
          },
        ]
      })
    }
  }

  mock_data "aws_partition" {
    defaults = {
      id         = "aws"
      partition  = "aws"
      dns_suffix = "amazonaws.com"
    }
  }

  mock_data "aws_subnet" {
    defaults = {
      vpc_id            = "vpc-0aaaaaaaaaaaaaaaa"
      availability_zone = "us-east-1a"
    }
  }

  mock_data "aws_ami" {
    defaults = {
      id = "ami-0aaaaaaaaaaaaaaaa"
    }
  }

  mock_data "aws_vpc_endpoint_service" {
    defaults = {
      service_name = "com.amazonaws.us-east-1.ssm"
    }
  }
}

# Inputs shared by every run below unless a run's own `variables` block
# overrides them. vpc_id matches the aws_subnet mock's default vpc_id above,
# so precondition 1 (main.tf:206-209) passes in every run except
# subnet_in_wrong_vpc_rejected, which deliberately overrides the mocked
# subnet's vpc_id to mismatch it.
variables {
  vpc_id          = "vpc-0aaaaaaaaaaaaaaaa"
  subnet_id       = "subnet-0bbbbbbbbbbbbbbbb"
  container_image = "groundkit:latest"
}

# Baseline: every precondition's happy path at once, with no embedding
# endpoint configured (the BM25-only default -- SPEC.md §10's local
# definition of done). Every other run in this file assumes this one plans
# clean; if it doesn't, the mocks or the module's non-precondition logic are
# broken, not the precondition coverage below.
run "valid_inputs_plan_clean" {
  command = plan

  assert {
    condition     = aws_security_group.instance.vpc_id == var.vpc_id
    error_message = "security group should plan into the requested VPC"
  }
}

# Precondition 2 (main.tf:211-214) is documented above as unreachable through
# the variable-validated path, so no run targets it with expect_failures
# against aws_security_group.instance. This run instead proves where
# "not-a-url" -- the input that raised the question -- actually gets
# rejected: variable validation (variables.tf:177-180), before the
# configuration ever reaches a plan, let alone this precondition.
run "embedding_base_url_invalid_url_rejected_at_variable_validation" {
  command = plan

  variables {
    embedding_base_url = "not-a-url"
  }

  expect_failures = [
    var.embedding_base_url,
  ]
}

# Precondition: main.tf:221-229 (`!local.embed_is_ipv6`).
#
# Matters because every egress rule this module writes is cidr_ipv4,
# including the standing HTTPS rule (main.tf:249-257) -- an IPv6 endpoint is
# unreachable regardless of port. The comment on this precondition records a
# real regression: the rejection used to ride on embed_needs_egress, which is
# false at port 443 because the standing rule already covers it, so an
# `https://[2001:db8::10]`-style endpoint on 443 slipped through with no rule
# created and every request timing out. A bracketed IPv6 literal on the
# default scheme/port passes variable validation cleanly, so this is exactly
# the case that regression needs a standing test for.
run "ipv6_embedding_endpoint_rejected" {
  command = plan

  variables {
    embedding_base_url = "https://[2001:db8::1]/"
  }

  expect_failures = [
    aws_security_group.instance,
  ]
}

# Precondition: main.tf:231-239 (`!local.embed_needs_egress ||
# local.embed_cidr != ""`).
#
# Matters because a DNS-named embedding host on a non-443 port (Ollama's
# default 11434 is the module's own documented case, main.tf:56-58) cannot be
# turned into a /32 at plan time -- nothing here resolves DNS. Without
# embedding_egress_cidr set, the module would otherwise accept the
# configuration, render `--dense` into the bootstrap script, create no egress
# rule for the endpoint, and let every embed call time out at the security
# group with nothing in the service's own logs to explain why.
run "dns_embedding_endpoint_without_cidr_rejected" {
  command = plan

  variables {
    embedding_base_url    = "http://embed.internal.example.com:11434"
    embedding_egress_cidr = ""
  }

  expect_failures = [
    aws_security_group.instance,
  ]
}

# Precondition: main.tf:206-209 (`data.aws_subnet.this.vpc_id ==
# var.vpc_id`).
#
# Matters because this is the constraint variables.tf's subnet_id
# description has always stated ("Must be in var.vpc_id") with nothing
# enforcing it until this precondition landed -- EC2 previously rejected the
# mismatch only at RunInstances, after the role, profile, volume and this
# security group already existed. override_data here is the proof this suite
# exists to provide: a mocked data source reaches a data-source-dependent
# precondition, not only the pure-locals ones above.
run "subnet_in_wrong_vpc_rejected" {
  command = plan

  override_data {
    target = data.aws_subnet.this
    values = {
      vpc_id            = "vpc-0mismatchedvpc00"
      availability_zone = "us-east-1a"
    }
  }

  expect_failures = [
    aws_security_group.instance,
  ]
}
