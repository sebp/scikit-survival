#!/bin/bash
set -xe

if [ "x$NO_SLOW" = "xfalse" ]
then
  bash <(curl -s https://codecov.io/bash) -f tests/coverage.xml
  pip install codacy-coverage
  python-codacy-coverage -r tests/coverage.xml
fi
