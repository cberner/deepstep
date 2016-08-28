.PHONY: all test coverage coverage_data

all: test

test:
	mypy --disallow-untyped-defs --silent-imports deepstep/ *.py tests/
	pylint --reports=n --max-line-length=120\
		--disable=missing-docstring\
		--disable=fixme\
		--disable=unused-import\
		--disable=too-many-statements\
		--disable=too-many-branches\
		--disable=too-many-locals\
		--disable=too-many-arguments\
		--disable=too-few-public-methods\
		--disable=duplicate-code\
		--disable=locally-disabled\
		deepstep/ *.py tests/*.py
	python -m unittest discover -s tests

coverage_data:
	coverage run --source=deepstep -m unittest discover -s tests

coverage: coverage_data
	coverage report
