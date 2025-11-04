.PHONY: format lint

format:
	black backend tests

lint:
	ruff check backend tests
