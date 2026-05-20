.PHONY: all smoke test audit-check download-small

PYTHON ?= .venv/bin/python

all: test smoke audit-check

test:
	$(PYTHON) -m pytest test_model.py -q

smoke:
	$(PYTHON) experiments/00_smoke/run.py

audit-check:
	$(PYTHON) -m compileall logstruct experiments

download-small:
	$(PYTHON) -c 'from experiments.shared.data_loader import download_source, write_checksums, prepare_uci_pancancer; paths = [download_source("uci_pancancer")]; write_checksums(paths); prepare_uci_pancancer()'
