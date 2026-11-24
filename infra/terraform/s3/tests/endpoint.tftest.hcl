mock_provider "aws" {}

run "accept_loopback_endpoint" {
  command = plan

  variables {
    endpoint = "http://127.0.0.1:5000"
  }
}

run "reject_non_loopback_endpoint" {
  command = plan

  variables {
    endpoint = "http://192.0.2.1:5000"
  }

  expect_failures = [var.endpoint]
}
