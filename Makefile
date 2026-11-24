FAULTWATCH_PORT ?= 8000

.PHONY: setup train demo replay test stop

setup:
	uv sync --locked

train:
	uv run --locked faultwatch fetch
	uv run --locked faultwatch train

demo:
	FAULTWATCH_PORT=$(FAULTWATCH_PORT) docker compose up --build --wait

replay:
	uv run --locked faultwatch replay --url http://127.0.0.1:$(FAULTWATCH_PORT)

test:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked pytest -q

stop:
	docker compose down

KIND ?= kind
KUBECTL = kubectl --kubeconfig .local/kubeconfig --context kind-faultwatch --namespace faultwatch

.PHONY: kind-up kind-forward kind-down

kind-up:
	KIND="$(KIND)" ./infra/kind/up.sh

kind-forward:
	$(KUBECTL) port-forward --address 127.0.0.1 service/api 18000:8000

kind-down:
	$(KIND) delete cluster --name faultwatch --kubeconfig .local/kubeconfig
