#!/bin/bash 

set -eux 

if [ ! -f analyse.py ]; then
    cd IMDB 
fi

python analyse.py
