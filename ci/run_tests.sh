#!/bin/bash
set -e
export PYTHONWARNINGS="default"

cd tests/
if [[ "x$NO_SLOW" != "xtrue" ]]; then
  nosetests --with-coverage --cover-xml --cover-package=sksurv --cover-tests
else
  nosetests -a "!slow" --with-coverage --cover-xml --cover-package=sksurv --cover-tests
fi
