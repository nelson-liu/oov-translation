#!/bin/bash
set -e

echo 'List files from cached directories'
if [ -d $HOME/download ]; then
    echo 'download:'
    ls $HOME/download
fi
if [ -d $HOME/.cache/pip ]; then
    echo 'pip:'
    ls $HOME/.cache/pip
fi

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Add the miniconda bin directory to $PATH
export PATH=/home/travis/miniconda3/bin:$PATH
echo $PATH

# Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -f /home/travis/miniconda3/bin/activate ]]
then
    if [[ ! -f miniconda.sh ]]
    then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
             -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -f
    conda update --yes conda
    echo "Creating environment to run tests in."
    conda create -q -n testenv --yes python="$PYTHON_VERSION"
fi
cd ..
popd

# Activate the python environment we created.
source activate testenv

# Install requirements via pip in our conda environment
pip install -r requirements.txt

# Install pytorch only if we are running tests
if [[ "$SKIP_TESTS" != "true" ]]; then
    conda install -q --yes pytorch=0.2 torchvision -c soumith
fi

# Download data that we need during tests
if [[ "$SKIP_TESTS" != "true" ]]; then
    pushd .
    cd
    mkdir -p oov_data
    cd oov_data
    # Amharic FastText vectors
    wget -c -N \
         https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.am.zip
    cd ..
    popd
fi

du -sch ~/.[!.]* * |sort -h
