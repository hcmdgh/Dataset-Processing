#!/bin/bash 

set -eux 

DIR=$(dirname $(readlink -f "$0"))
cd $DIR 

python analyse.py 
