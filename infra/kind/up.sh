#!/bin/sh
set -eu

cd "$(dirname "$0")/../.."
kind=${KIND:-kind}
kubeconfig="$PWD/.local/kubeconfig"
model_dir=${FAULTWATCH_MODEL_DIR:-artifacts/fd001-v1}
if [ ! -f "$model_dir/manifest.json" ]; then
    printf '%s\n' "Run make train before starting kind." >&2
    exit 1
fi
model_dir=$(cd "$model_dir" && pwd)
mkdir -p .local

if "$kind" get clusters | grep -Fxq faultwatch; then
    "$kind" export kubeconfig --name faultwatch --kubeconfig "$kubeconfig"
    existing=yes
else
    "$kind" create cluster --name faultwatch --kubeconfig "$kubeconfig" --wait 120s \
        --image kindest/node:v1.34.3@sha256:08497ee19eace7b4b5348db5c6a1591d7752b164530a36f855cb0f2bdcbadd48 \
        --config - <<CONFIG
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
networking:
  apiServerAddress: "127.0.0.1"
nodes:
  - role: control-plane
    extraMounts:
      - hostPath: "$model_dir"
        containerPath: /faultwatch-model
        readOnly: true
CONFIG
    existing=no
fi

docker compose build api
docker image inspect postgres:17.12-alpine >/dev/null 2>&1 || docker pull postgres:17.12-alpine
platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' faultwatch:local)
archive=$(mktemp "${TMPDIR:-/tmp}/faultwatch-images.XXXXXX")
trap 'rm -f "$archive"' EXIT
docker image save --platform "$platform" --output "$archive" faultwatch:local postgres:17.12-alpine
"$kind" load image-archive --name faultwatch "$archive"

kube() {
    kubectl --kubeconfig "$kubeconfig" --context kind-faultwatch "$@"
}

kube create namespace faultwatch --dry-run=client -o yaml | kube apply -f -
kube -n faultwatch apply -f infra/kind/stack.yaml
kube -n faultwatch rollout status deployment/postgres --timeout=120s
if [ "$existing" = yes ]; then
    kube -n faultwatch rollout restart deployment/api
fi
kube -n faultwatch rollout status deployment/api --timeout=120s
