PY   ?= .venv/bin/python
NRAG ?= .venv/bin/nrag
ARGS ?=

.PHONY: setup ingest extract score index eval app test

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

ingest:
	$(NRAG) ingest $(ARGS)

extract:
	$(NRAG) extract $(ARGS)

score:
	$(NRAG) score $(ARGS)

index:
	$(NRAG) index $(ARGS)

eval:
	$(NRAG) eval $(ARGS)

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -m "not network" $(ARGS)
