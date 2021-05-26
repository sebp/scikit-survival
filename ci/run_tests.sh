#!/bin/bash
set -e
export PYTHONWARNINGS="default"

cd tests/

coverage_opts=(--cov-config=../.coveragerc --cov=sksurv --cov-report xml --cov-report term-missing)
pytest_opts=(--strict-markers)

if [ "x${NO_SLOW:-false}" != "xtrue" ]; then
  coverage erase
  rm -f coverage.xml
else
  pytest_opts+=(-m 'not slow')
fi

pytest "${pytest_opts[@]}" "${coverage_opts[@]}"
