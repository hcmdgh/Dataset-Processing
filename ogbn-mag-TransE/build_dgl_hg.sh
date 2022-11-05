#!/bin/bash 

set -eux 

if [ ! -f build_dgl_hg.py ]; then
    cd ogbn-mag-TransE 
fi

python build_dgl_hg.py
