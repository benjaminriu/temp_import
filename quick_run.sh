#!/bin/bash

for (( i=0; i<2; i++ )); do
    python ./temp_run_all.py --train_size 30 --method RF --dataset_id $i --avoid_duplicates FALSE
done