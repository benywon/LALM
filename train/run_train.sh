#!bin/bash

for i in {0..9}; do
    python3.6 -m torch.distributed.launch --nproc_per_node=8 train.py --num=$i > log/$i.txt
done