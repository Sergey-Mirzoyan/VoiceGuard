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

.PHONY: all install test lint fmt check-ffmpeg docker-base

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
