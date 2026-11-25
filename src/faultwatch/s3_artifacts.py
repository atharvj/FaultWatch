"""Transport a small, verified model bundle through a loopback S3 endpoint."""

from contextlib import closing
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import boto3
from botocore.config import Config

from faultwatch.inference import MODEL_FILES, load_manifest

MAX_FILE_BYTES = 2 * 1024 * 1024


def local_s3(endpoint: str):
    address = urlsplit(endpoint)
    if (
        address.scheme != "http"
        or address.hostname != "127.0.0.1"
        or not address.port
        or address.username is not None
        or address.path
        or address.query
        or address.fragment
    ):
        raise ValueError("S3 transport requires an HTTP loopback endpoint")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="faultwatch-local",
        # The loopback emulator accepts dummy credentials.
        aws_secret_access_key="faultwatch-local",  # noqa: S106
        config=Config(proxies={}, connect_timeout=3, read_timeout=3),
    )


def publish_bundle(client, bucket: str, bundle: Path) -> str:
    manifest = load_manifest(bundle)
    selected = manifest["selected_model"]
    model_file = MODEL_FILES[selected]
    artifacts = {
        "manifest.json": (bundle / "manifest.json").read_bytes(),
        model_file: (bundle / model_file).read_bytes(),
    }
    if any(len(content) > MAX_FILE_BYTES for content in artifacts.values()):
        raise ValueError("Artifact exceeds the 2 MiB limit")
    if sha256(artifacts[model_file]).hexdigest() != manifest["artifact_sha256"][selected]:
        raise ValueError("Model artifact checksum mismatch")
    prefix = f"models/{manifest['model_version']}/{sha256(artifacts['manifest.json']).hexdigest()}"
    for name in (model_file, "manifest.json"):
        client.put_object(Bucket=bucket, Key=f"{prefix}/{name}", Body=artifacts[name])
    return prefix


def _get(client, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    with closing(response["Body"]) as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Artifact exceeds the 2 MiB limit")
    return content


def restore_bundle(client, bucket: str, prefix: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory)
        content = _get(client, bucket, f"{prefix}/manifest.json")
        if sha256(content).hexdigest() != prefix.rsplit("/", 1)[-1]:
            raise ValueError("Manifest checksum mismatch")
        (staging / "manifest.json").write_bytes(content)
        manifest = load_manifest(staging)
        selected = manifest["selected_model"]
        name = MODEL_FILES[selected]
        model = _get(client, bucket, f"{prefix}/{name}")
        if sha256(model).hexdigest() != manifest["artifact_sha256"][selected]:
            raise ValueError("Model artifact checksum mismatch")
        (staging / name).write_bytes(model)
        staging.chmod(0o755)
        staging.rename(output)
