.PHONY: clean lint data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using black and flake8
lint:
	black src && flake8 src

data:
	touch data/raw/example.txt

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

