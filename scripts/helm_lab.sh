#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
helm=${HELM:-helm}
chart=infra/helm/faultwatch
release="faultwatch-helm-test-$$"
kubeconfig=.local/kubeconfig
context=kind-faultwatch
namespace=faultwatch

release_command() {
    "$helm" "$@" --kubeconfig "$kubeconfig" --kube-context "$context" --namespace "$namespace"
}

kube() {
    kubectl --kubeconfig "$kubeconfig" --context "$context" --namespace "$namespace" "$@"
}

"$helm" lint "$chart"
"$helm" template "$release" "$chart" >/dev/null
trap 'release_command uninstall "$release" --ignore-not-found >/dev/null' EXIT
release_command install "$release" "$chart" --wait --timeout 120s
readiness() {
    kube exec "deploy/$release-api" -c api -- python -c 'from urllib.request import urlopen; assert urlopen("http://127.0.0.1:8000/readyz").status == 200'
}
readiness
release_command upgrade "$release" "$chart" --set resources.requests.cpu=300m --wait --timeout 120s
readiness
request=$(kube get "deploy/$release-api" -o jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}')
test "$request" = 300m
release_command history "$release" -o json
