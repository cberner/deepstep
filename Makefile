all: test

test:
	pylint --reports=n --max-line-length=120 --disable=missing-docstring,fixme deepstep.py
	mypy --silent-imports deepstep.py
