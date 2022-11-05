#!/bin/bash 

set -eux 

if [ ! -f build_dgl_hg.py ]; then
    cd ogbn-mag
fi

python build_dgl_hg.py
