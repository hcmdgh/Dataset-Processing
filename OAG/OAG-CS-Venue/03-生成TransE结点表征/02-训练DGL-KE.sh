#!/bin/bash 

eval "$(conda shell.bash hook)"

conda activate py37 

set -eux 

emb_dim=400 

cd TransE_emb 

rm -rf ckpts entities.tsv relations.tsv 

DGLBACKEND=pytorch dglke_train \
    --model TransE_l2 \
    --batch_size 1000 \
    --neg_sample_size 200 \
    --hidden_dim $emb_dim \
    --gamma 10 \
    --lr 0.1 \
    --max_step 400000 \
    --log_interval 10000 \
    -adv \
    --gpu 0 \
    --regularization_coef 1e-9 \
    --data_path ./ \
    --data_files "./triplets.tsv" \
    --format raw_udd_hrt \
    --dataset dataset
