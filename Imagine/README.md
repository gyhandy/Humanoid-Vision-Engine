# HVE-Imagine

### 1. Train the Model

- In terminal, run `cd Imagine`

- Edit `script.sh`. Set the path of dataset and output file.


```bash
#!/usr/bin/env bash

FEATURE=texture # choose from texture, color, shape

python main.py --cuda 0, 1 \
               --mode train \
               --batch_size 16 \
               --dataset_path /lab/tmpig8d/u/yao_data/human_simulation_engine/V3_${FEATURE}_dataset \
               --output_path out/deeper/${FEATURE}
```      
- run `sh script.sh`

### 2. Run Generation

- Edit `test.sh`. 
    - Set the path of dataset and output file.
    - Set the test checkpoint file
    - If want to use the mismatch shape, texture, color as input, set `--mismatch`
    - Example
    
```bash
FEATURE=texture # choose from texture, color, shape

python main.py --cuda 0 \
               --mode predict \
               --batch_size 16 \
               --dataset_path /lab/tmpig8d/u/yao_data/human_simulation_engine/V3_${FEATURE}_dataset \
               --output_path out/deeper/${FEATURE} \
               --test_epoch 269 \

```

- run `sh test.sh`


### 3. Calculate the FID

- Install [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

```bash
pip install pytorch-fid
```

- Resize the groughtruth image

For example:

```bash
#!/usr/bin/env bash
FEATURE=texture # choose from texture, color, shape

# Ground truth images dir
dataset_path=/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_${FEATURE}_dataset/ori/valid/

# Processed gt images dir
process_path=out/deeper_deeper_res_new_texture/${FEATURE}/gt

python create_dataset.py --ori_path ${dataset_path} --path ${process_path}
```

- Run Fid code:


```bash
#!/usr/bin/env bash
FEATURE=texture # choose from texture, color, shape

# Processed gt images dir
process_path=out/deeper_deeper_res_new_texture/${FEATURE}/gt

# Generation result dir
output_path=out/deeper_deeper_res_new_texture/${FEATURE}/result_mismatch

python -m pytorch_fid ${process_path} ${output_path} --device cuda:1 --batch-size 128
```
