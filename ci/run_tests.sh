#!/bin/bash
set -e

if [[ "x$NO_SLOW" != "xtrue" ]]; then
  nosetests -w tests --with-coverage --cover-xml --cover-package=sksurv --cover-tests
else
  nosetests -w tests -a "!slow" --with-coverage --cover-xml --cover-package=sksurv --cover-tests
fi
