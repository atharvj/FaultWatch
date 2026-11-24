terraform {
  required_version = "~> 1.17.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "= 5.100.0"
    }
  }
}

variable "endpoint" {
  type        = string
  description = "Loopback Moto S3 endpoint."
  default     = "http://127.0.0.1:5000"

  validation {
    condition     = can(regex("^http://127[.]0[.]0[.]1:[1-9][0-9]{0,4}$", var.endpoint))
    error_message = "The S3 endpoint must be an HTTP loopback address with a port."
  }
}

provider "aws" {
  region                      = "us-east-1"
  access_key                  = "faultwatch-local"
  secret_key                  = "faultwatch-local"
  s3_use_path_style           = true
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_region_validation      = true
  skip_requesting_account_id  = true
  endpoints {
    s3 = var.endpoint
  }
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "faultwatch-local-artifacts"
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "bucket" {
  value = aws_s3_bucket.artifacts.id
}
