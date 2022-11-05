#!/bin/bash 

source /home/gh/home/Anaconda/etc/profile.d/conda.sh
conda activate py37 

set -eux 

if [ ! -f preprocess.py ]; then
    cd OAG-from-PTHGNN 
fi

python -m debugpy --listen localhost:14285 --wait-for-client preprocess.py 
