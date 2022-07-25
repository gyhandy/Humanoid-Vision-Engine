#!/usr/bin/env bash

FEATURE=texture

python main.py --cuda 0,1 \
               --batch_size 16 \
               --dataset_path /lab/tmpig8d/u/yao_data/human_simulation_engine/V3_${FEATURE}_dataset \
               --output_path out/deeper/${FEATURE}
