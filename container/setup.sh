#!/bin/env bash
# NOTE: This is an example on how to setup scripts which are sourced when activating conda environments
# The script refers to ${CONDA_PREFIX} which is set when a conda environment is activated
# typical use cases are to setup environment variables which are specific to an environment and are supposed to be set
# upon activation and removed when the environment is deactivated
# YOU CAN DELETE THIS SCRIPT AND THE CONTAINING FOLDER IF YOU HAVE NO USE FOR THIS
# creating startup and shutdown scripts for conda environments which are executed when (de)activating conda environments
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
echo 'echo "activation for conda environment in ${CONDA_PREFIX}"' > ${CONDA_PREFIX}/etc/conda/activate.d/hqs_python_template.sh
echo 'echo "installed jq version: $(jq --version)"' >> ${CONDA_PREFIX}/etc/conda/activate.d/hqs_python_template.sh
echo 'echo "deactivation for conda environment in ${CONDA_PREFIX}"' > ${CONDA_PREFIX}/etc/conda/deactivate.d/hqs_python_template.sh
echo 'echo "The answer to everything is: $(get_answer)"' >> ${CONDA_PREFIX}/etc/conda/deactivate.d/hqs_python_template.sh
