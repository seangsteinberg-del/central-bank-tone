# Developer and demo shortcuts. See README.md for the manual (Windows/PowerShell) equivalents.
.PHONY: help sync gate test lint db-up db-down migrate eval eval-cross train chart serve demo demo-lite live

help:
	@echo "sync       - install the workspace (uv sync)"
	@echo "gate       - run the full quality gate (ruff, mypy, imports, tests)"
	@echo "eval       - score classifier vs lexicon vs the FOMC benchmark, plot tone vs rates"
	@echo "eval-cross - out-of-distribution check: score the classifier on op-fed (MIT)"
	@echo "train      - retrain the supervised tone-model artifact"
	@echo "demo-lite  - serve the keyless UI with NO key and NO Docker (SQLite + offline model); starts empty"
	@echo "live       - real Gemini + native PostgreSQL, NO Docker/pgvector: ingest real speeches and serve"
	@echo "db-up      - start Postgres + pgvector (docker compose)"
	@echo "migrate    - apply migrations to head"
	@echo "serve      - run the web UI (uvicorn, reload)"
	@echo "demo       - db-up + migrate + serve (needs Docker; set CBT_DATABASE_URL/.env first)"

sync:
	uv sync

gate:
	uv run ruff check . && uv run ruff format --check . && uv run mypy && \
	uv run python scripts/check_imports.py && uv run pytest -m "not llm"

lint:
	uv run ruff check . && uv run ruff format --check .

test:
	uv run pytest -m "not llm"

eval:
	uv run python scripts/eval_tone.py
	uv run python scripts/tone_trajectory.py

eval-cross:
	uv run python scripts/eval_cross_dataset.py

train:
	uv run python scripts/train_tone_model.py

demo-lite:
	uv run python scripts/run_demo.py

live:
	uv run python scripts/run_live.py

db-up:
	docker compose up -d db

db-down:
	docker compose down

migrate:
	uv run python scripts/migrate.py

serve:
	uv run uvicorn --factory cbt_web.app:create_app --reload

demo: db-up
	@echo "Waiting for Postgres to be ready ..."
	@until docker compose exec -T db pg_isready -U cbt -d cbt >/dev/null 2>&1; do sleep 1; done
	uv run python scripts/migrate.py
	uv run uvicorn --factory cbt_web.app:create_app
