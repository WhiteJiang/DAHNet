#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py --dataset nabirds --root /home/jx/data/nabirds --max-epoch 30 --batch-size 16 --max-iter 20 --code-length 12 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 15 --num-samples 4000 --info 'NAbirds' --momen 0.91