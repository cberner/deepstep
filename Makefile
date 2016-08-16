all: test

test:
	mypy --disallow-untyped-defs --silent-imports deepstep/ generate.py hyperflow.py
	pylint --reports=n --max-line-length=120\
		--disable=missing-docstring\
		--disable=fixme\
		--disable=too-many-locals\
		--disable=too-many-arguments\
		--disable=too-few-public-methods\
		deepstep/ generate.py hyperflow.py
