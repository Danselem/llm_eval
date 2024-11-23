.PHONY: clean dirs virtualenv lint requirements push pull reproduce

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create virtualenv.
## Activate with the command:
## source env/bin/activate
virtualenv:
	$(PYTHON_INTERPRETER) -m venv .venv
	$(info "Activate with the command 'source .venv/bin/activate'")

## Install Python Dependencies.
## Make sure you activate the virtualenv first!
requirements:
	# $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt