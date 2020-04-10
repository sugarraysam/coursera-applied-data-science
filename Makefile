export IPYTHONDIR = $(PWD)/.ipython

.PHONY: jupyter
jupyter: ## start jupyter server
	@pipenv run jupyter-notebook;

.PHONY: ipython
ipython: ## start ipython interpreter with preloaded modules
	@pipenv run ipython;

.PHONY: help
help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
