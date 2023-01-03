#!/bin/bash

# itterate throuhg 11

for weight_idx in {0..3}
do
    for img_num in {0..0}
    do
        echo $i $j
        python3 inference_flax_many_prompts.py  $weight_idx  $img_num
    done
done
