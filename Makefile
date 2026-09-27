# .pth files in .venv may get the macOS "hidden" flag and be skipped by Python
export PYTHONPATH := $(CURDIR)/src


UV := $(shell which uv 2>/dev/null)

ifeq ($(UV),)
PYTHON := python3
PYTEST := pytest
RUFF := ruff
MYPY := mypy
SYNC := pip install -e .
else
PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
MYPY := uv run mypy
SYNC := uv sync --all-groups
endif

.PHONY: all install test lint fmt check-ffmpeg docker-base data stage0 demo passport

all: lint test

install:
	$(SYNC)

test:
	$(PYTEST) -q

lint:
	$(RUFF) check .
	$(MYPY) src

fmt:
	$(RUFF) format .

check-ffmpeg:
	$(PYTHON) scripts/check_ffmpeg.py

docker-base:
	docker build -f docker/Dockerfile.base -t voiceguard-base:latest .

# Download dataset and build manifest
data:
	$(PYTHON) -m voiceguard.data

# Run stage 0 experiment (train D1, build D2 refs, evaluate, report)
stage0:
	$(PYTHON) -m voiceguard.experiments

# Build passport from stage0 results
passport:
	$(PYTHON) -m voiceguard.fusion.build_passport

# Run the demo server
demo:
	uv run uvicorn voiceguard.engine.api:app --port 8000
