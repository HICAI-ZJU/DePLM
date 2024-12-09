#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

CUDA_VISIBLE_DEVICES=0 python src/train.py data=batched_proteingym data.assay_index=196 data.split_index=0 data.split_type=random task_name=test model.model=esm2 model.diff_total_step=3 debug=default