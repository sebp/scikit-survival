#!/bin/sh
set -xe

if [ "x$NO_SLOW" = "xfalse" ]
then
  bash <(curl -s https://codecov.io/bash)
  pip install codacy-coverage
  python-codacy-coverage -r coverage.xml
fi
