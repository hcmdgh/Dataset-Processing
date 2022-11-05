#!/bin/bash 

set -eux 

if [ ! -f build_dgl_hg.py ]; then
    cd IMDB 
fi

python build_dgl_hg.py
