#!/bin/bash 

set -eux 

DIR=$(dirname $(readlink -f "$0"))
cd $DIR 

python 02-构建异构图.py 
