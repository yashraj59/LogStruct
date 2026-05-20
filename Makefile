.PHONY: all smoke test audit-check download-small metabric-1000 gtex-500 norman-500 tabula-500

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

metabric-1000:
	$(PYTHON) experiments/01_brca_pam50/run.py --max-genes 1000 --methods dummy_most_frequent logistic_l2 nslr_string_700 logstruct_string_700 --device cuda
	$(PYTHON) experiments/01_brca_pam50/make_outputs.py

gtex-500:
	$(PYTHON) experiments/04_gtex_tissue/run.py --max-genes 500 --methods dummy_most_frequent logistic_l2 logstruct_identity logstruct_string_700 --device cuda
	$(PYTHON) experiments/04_gtex_tissue/make_outputs.py

norman-500:
	$(PYTHON) experiments/06_norman_perturb_identity/run.py --max-genes 500 --methods dummy_most_frequent logistic_l2 logstruct_identity logstruct_string_700 --device cuda
	$(PYTHON) experiments/06_norman_perturb_identity/make_outputs.py

tabula-500:
	$(PYTHON) experiments/05_tabula_sapiens_celltype/run.py --max-genes 500 --min-cells 500 --max-cells-per-class 3000 --methods dummy_most_frequent logstruct_identity logstruct_string_700 --device cuda
	$(PYTHON) experiments/05_tabula_sapiens_celltype/make_outputs.py
