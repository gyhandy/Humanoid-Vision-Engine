# HVE-Imagine

### 1. To Train the model

- run `cd Imagine`

- Edit `script.sh`. Set the path of dataset and output file.


```bash
#!/usr/bin/env bash

FEATURE=texture # choose from texture, color, shape

python main.py --cuda 0,1 \
               --batch_size 16 \
               --dataset_path /lab/tmpig8d/u/yao_data/human_simulation_engine/V3_${FEATURE}_dataset \
               --output_path out/deeper/${FEATURE}
```      
- run `sh script.sh`
