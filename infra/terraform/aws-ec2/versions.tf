terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # A range, not a pin. A module that pins a patch version fights every
      # consumer that has a lockfile of its own; the ceiling guards the next
      # major, which is where resource schemas break.
      version = ">= 5.40.0, < 7.0.0"
    }
  }
}
