terraform {
  required_version = "~> 1.17.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "= 5.100.0"
    }
  }
}

variable "account_id" {
  type        = string
  description = "Account ID verified by the Free plan preflight."

  validation {
    condition     = can(regex("^[0-9]{12}$", var.account_id))
    error_message = "The account ID must contain twelve digits."
  }
}

variable "region" {
  type        = string
  description = "Region assigned to the AWS project."
}

variable "bucket_name" {
  type        = string
  description = "Dedicated bucket name for this model artifact deployment."

  validation {
    condition     = can(regex("^faultwatch-model-[0-9a-f]{16}$", var.bucket_name))
    error_message = "Use faultwatch-model- followed by 16 lowercase hexadecimal characters."
  }
}

provider "aws" {
  region                  = var.region
  allowed_account_ids     = [var.account_id]
  skip_metadata_api_check = true
}

resource "aws_s3_bucket" "model" {
  bucket        = var.bucket_name
  force_destroy = false

  tags = {
    Project = "FaultWatch"
  }
}

resource "aws_s3_bucket_public_access_block" "model" {
  bucket                  = aws_s3_bucket.model.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model" {
  bucket = aws_s3_bucket.model.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "model_bucket" {
  value = aws_s3_bucket.model.bucket
}
