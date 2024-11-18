#!/bin/bash
# Simple init script for Python on DTU HPC
# Patrick M. Jensen, patmjen@dtu.dk, 2022

# Configuration
# This is what you should change for your setup
VENV_NAME=envDML    # Name of your virtualenv (default: venv)
VENV_DIR=.             # Where to store your virtualenv (default: current directory)
PYTHON_VERSION=3.11.9  # Python version (default: 3.9.14)
CUDA_VERSION=11.8      # CUDA version (default: 11.6)

# Load modules
module load python3/$PYTHON_VERSION
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" | grep "numpy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" | grep "scipy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" | grep "matplotlib/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" | grep "pandas/")
module load cuda/$CUDA_VERSION
CUDNN_MOD=$(module avail -o modulepath -t cudnn | grep "cuda-${CUDA_VERSION}" | sort | tail -n1)
if [[ ${CUDNN_MOD} ]]
then
    module load ${CUDNN_MOD}
fi

# Create virtualenv if needed and activate it
if [ ! -d "${VENV_DIR}/${VENV_NAME}" ]
then
    echo INFO: Did not find virtualenv. Creating...
    virtualenv "${VENV_DIR}/${VENV_NAME}"
fi
source "${VENV_DIR}/${VENV_NAME}/bin/activate"

# Select least used GPU if any are available
if command -v nvidia-smi &> /dev/null
then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.gpu,index --format=csv,noheader,nounits | sort -V | awk '{print $NF}' | head -n1)
    echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
fi
