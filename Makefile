.PHONY: format lint

format:
uv run --extra dev black backend tests

lint:
uv run --extra dev ruff check backend tests
