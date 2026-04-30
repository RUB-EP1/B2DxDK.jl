#!/bin/bash

ENV_NAME=tf-pwa-env
PYTHON_VERSION=3.9
PACKAGE_PATH=tf-pwa

source $(conda info --base)/etc/profile.d/conda.sh

conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

conda install -y -c conda-forge \
    numpy<1.25.0 scipy matplotlib jupyterlab ipython pandas tqdm h5py numba git pip

pip install --upgrade pip setuptools wheel
pip install "tensorflow>=2.7" tensorflow-probability pyyaml graphviz

cd $PACKAGE_PATH
pip install -e .

echo "Environment $ENV_NAME is ready."
