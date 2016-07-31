all: test

test:
	mypy --silent-imports deepstep.py
	pylint --reports=n --max-line-length=120 --disable=missing-docstring,fixme,too-many-locals deepstep.py
