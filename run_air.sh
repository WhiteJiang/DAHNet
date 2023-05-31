#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py --dataset aircraft --root /storage/jx_data/ --max-epoch 30 --batch-size 16 --max-iter 20 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 15 --num-samples 2000 --info 'Aircraft' --momen 0.91