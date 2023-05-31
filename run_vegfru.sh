#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py --dataset vegfru --root /storage/jx_data/vegfru/ --max-epoch 30 --batch-size 16 --max-iter 20 --code-length 48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 15 --num-samples 4000 --info 'VegFru' --momen 0.91