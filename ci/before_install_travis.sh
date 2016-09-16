#!/bin/bash
set -xe

MINICONDA_DIR="$HOME/miniconda3"

if [ -d "$MINICONDA_DIR" ] && [ -e "$MINICONDA_DIR/bin/conda" ]
then
  echo "Miniconda install already present from cache: $MINICONDA_DIR"
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
else
  echo "Installing Miniconda"
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b -f -p "$MINICONDA_DIR"

  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda install anaconda-client
  conda create -n ssvm python=$TRAVIS_PYTHON_VERSION nose coverage
fi

# The next couple lines fix a crash with multiprocessing on Travis and are not specific to using Miniconda
sudo rm -rf /dev/shm
sudo ln -s /run/shm /dev/shm
