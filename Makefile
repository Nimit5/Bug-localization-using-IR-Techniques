# define the name of the virtual environment directory
VIRTUAL_PROJ := venv

# default target, when make executed without arguments
all: venv

$(VIRTUAL_PROJ)/bin/activate: requirements.txt
	python3 -m pip install -U pip setuptools
	python3 -m venv $(VIRTUAL_PROJ)
	pip install -r requirements.txt
	python3 -m spacy download en_core_web_lg

# venv is a shortcut target
venv: $(VIRTUAL_PROJ)/bin/activate

run: venv
	python3 master/main.py

clean:
	rm -rf $(VIRTUAL_PROJ)
	find . -type f -name '*.pyc' -delete

.PHONY: all venv run clean
