#!/bin/bash 

set -eux 

source /home/gh/home/Anaconda/etc/profile.d/conda.sh
conda activate py37

DIR=$(dirname $(readlink -f "$0"))
cd $DIR 

python 01-抽取边关系.py 
