#!/bin/bash

conda activate seg_env

python3 check_data.py --input_file $1 --output_file $2
