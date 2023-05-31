#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run.py --dataset food101 --root /storage/jx_data/food-101 --max-epoch 30 --batch-size 16 --max-iter 40 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 25 --num-samples 2000 --info 'Food101' --momen 0.91
