output "instance_id" {
  description = "EC2 instance id. The target of an SSM session."
  value       = aws_instance.this.id
}

output "instance_private_ip" {
  description = <<-EOT
    Private address of the instance. Informational only: nothing listens on it.
    The service publishes to the instance's loopback interface and the security
    group has no ingress rules (ADR-0020 decision 2).
  EOT
  value       = aws_instance.this.private_ip
}

output "data_volume_id" {
  description = <<-EOT
    EBS volume holding the index. It has `prevent_destroy` set and outlives
    instance replacement — but it has NO backups: this module creates no
    snapshot schedule and no DLM policy, and the volume holds document text
    rather than a rebuildable index (SPEC.md §7).
  EOT
  value       = aws_ebs_volume.data.id
}

output "security_group_id" {
  description = "Instance security group. It has no ingress rules, by design."
  value       = aws_security_group.instance.id
}

output "iam_role_arn" {
  description = "Instance role. Its only attached policy is AmazonSSMManagedInstanceCore."
  value       = aws_iam_role.this.arn
}

output "ssm_port_forward_command" {
  description = <<-EOT
    The way in. There is no other one: no public address, no ingress rule, no
    SSH key. Run this, then reach the service at http://127.0.0.1:<port> on your
    own machine. Requires the Session Manager plugin and `ssm:StartSession` on
    this instance.
  EOT
  value = join(" ", [
    "aws ssm start-session",
    "--target ${aws_instance.this.id}",
    "--document-name AWS-StartPortForwardingSession",
    "--parameters '{\"portNumber\":[\"${var.host_port}\"],\"localPortNumber\":[\"${var.host_port}\"]}'",
  ])
}
