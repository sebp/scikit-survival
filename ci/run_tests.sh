#!/bin/bash
set -e
export PYTHONWARNINGS="default"

pytest_opts=("")

if [ "x${CI_NO_SLOW:-false}" != "xtrue" ]; then
  coverage erase
  rm -f coverage.xml
else
  pytest_opts+=(-m 'not slow')
fi

pytest --doctest-modules --doctest-continue-on-failure --pyargs sksurv

coverage run -m pytest "${pytest_opts[@]}"

coverage xml
coverage report
