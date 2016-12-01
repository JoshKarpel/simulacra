#!/bin/bash

# untar my Python installation and simulation package
tar -xzf python.tar.gz
tar -xzf compy.tar.gz
tar -xzf ionization.tar.gz

# make sure the script will use my Python installation
export PATH=$(pwd)/python/bin:$PATH
export LDFLAGS="-L$(pwd)/python/lib $LDFLAGS"

# run the python script
python run_sim.py $1
