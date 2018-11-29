#!/bin/bash
set -e
export PYTHONWARNINGS="default"

cd tests/
rm -f coverage.xml .coverage*

pytest_opts="--cov-config=../.coveragerc --cov=sksurv --cov-report xml --cov-report term-missing"

if [[ "x$NO_SLOW" != "xtrue" ]]; then
  py.test ${pytest_opts}
else
  py.test -m "not slow" ${pytest_opts}
fi
