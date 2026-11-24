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
