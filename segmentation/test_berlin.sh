#!/bin/bash

echo "current environment: $CONDA_DEFAULT_ENV"
#2ve seg_env

export CUDA_VISIBLE_DEVICES=0

python3 main.py --input_file '../../../ste/rnd/User/yusuf/city_data/Berlin_Summer' --input_channels 2 --output_channels 14 --epochs 5 --eval
