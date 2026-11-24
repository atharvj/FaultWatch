"""Provision a loopback bucket, round-trip the fitted model, and tear it down."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from moto.server import ThreadedMotoServer

from faultwatch.inference import load_predictor
from faultwatch.s3_artifacts import local_s3, publish_bundle, restore_bundle

ROOT = Path(__file__).resolve().parents[1]


def run(terraform: str, bundle: Path) -> dict:
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        _, port = server.get_host_and_port()
        endpoint = f"http://127.0.0.1:{port}"
        with TemporaryDirectory() as name:
            root = Path(name)
            shutil.copytree(
                ROOT / "infra/terraform/s3",
                root / "tf",
                ignore=shutil.ignore_patterns(".terraform"),
            )
            env = {**os.environ, "TF_VAR_endpoint": endpoint, "AWS_EC2_METADATA_DISABLED": "true"}

            def tf(*args):
                return subprocess.check_output(
                    [terraform, *args], cwd=root / "tf", env=env, text=True, timeout=300
                ).strip()

            tf("init", "-input=false")
            tf("validate")
            tf("plan", "-input=false", "-out=lab.tfplan")
            plan = json.loads(tf("show", "-json", "lab.tfplan"))
            resources = {
                entry["type"]
                for entry in plan["resource_changes"]
                if entry["change"]["actions"] == ["create"]
            }
            expected = {
                "aws_s3_bucket",
                "aws_s3_bucket_public_access_block",
                "aws_s3_bucket_server_side_encryption_configuration",
            }
            if resources != expected:
                raise RuntimeError(f"Unexpected create plan: {resources}")
            client = local_s3(endpoint)
            try:
                tf("apply", "-input=false", "lab.tfplan")
                bucket = tf("output", "-raw", "bucket")
                protection = client.get_public_access_block(Bucket=bucket)[
                    "PublicAccessBlockConfiguration"
                ]
                assert len(protection) == 4 and all(protection.values())
                rules = client.get_bucket_encryption(Bucket=bucket)[
                    "ServerSideEncryptionConfiguration"
                ]["Rules"]
                assert rules[0]["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256"
                prefix = publish_bundle(client, bucket, bundle)
                restore_bundle(client, bucket, prefix, root / "restored")
                original = load_predictor(bundle)
                restored = load_predictor(root / "restored")
                assert restored.artifact_sha256 == original.artifact_sha256
                result = {
                    "created_resources": sorted(resources),
                    "public_access_block": protection,
                    "encryption": "AES256",
                    "model": restored.model_name,
                    "model_checksum_verified": True,
                    "objects_transferred": 2,
                }
            finally:
                for item in client.list_buckets().get("Buckets", []):
                    bucket = item["Name"]
                    for obj in client.list_objects_v2(Bucket=bucket).get("Contents", []):
                        client.delete_object(Bucket=bucket, Key=obj["Key"])
                tf("destroy", "-auto-approve", "-input=false")
            result["buckets_after_destroy"] = len(client.list_buckets().get("Buckets", []))
            assert result["buckets_after_destroy"] == 0
            return result
    finally:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terraform", default="terraform")
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/fd001-v1"))
    args = parser.parse_args()
    executable = shutil.which(args.terraform)
    if not executable:
        parser.error("Terraform executable not found")
    print(json.dumps(run(str(Path(executable).resolve()), args.bundle), indent=2))
