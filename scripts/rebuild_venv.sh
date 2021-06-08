# /bin/bash

pipenv --rm
pipenv update --dev
pipenv lock -r > requirements.txt
pipenv lock --dev -r > requirements-dev.txt
