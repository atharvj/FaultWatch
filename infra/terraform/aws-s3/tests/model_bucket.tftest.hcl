mock_provider "aws" {}

variables {
  account_id  = "123456789012"
  region      = "us-east-1"
  bucket_name = "faultwatch-model-0123456789abcdef"
}

run "plan_private_encrypted_bucket" {
  command = plan

  assert {
    condition     = output.model_bucket == var.bucket_name && !aws_s3_bucket.model.force_destroy
    error_message = "The plan must use the dedicated bucket without forced deletion."
  }

  assert {
    condition = alltrue([
      aws_s3_bucket_public_access_block.model.block_public_acls,
      aws_s3_bucket_public_access_block.model.block_public_policy,
      aws_s3_bucket_public_access_block.model.ignore_public_acls,
      aws_s3_bucket_public_access_block.model.restrict_public_buckets,
    ])
    error_message = "Every public access block must be enabled."
  }

  assert {
    condition     = one(aws_s3_bucket_server_side_encryption_configuration.model.rule).apply_server_side_encryption_by_default[0].sse_algorithm == "AES256"
    error_message = "The bucket must use SSE-S3 by default."
  }
}

run "reject_unrelated_bucket" {
  command = plan

  variables {
    bucket_name = "existing-unrelated-bucket"
  }

  expect_failures = [var.bucket_name]
}
