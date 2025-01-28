#!/usr/bin/env bash

echo "PYTHONPATH=$PYTHONPATH"

# Example script to run pre-training on one machine with accelerate
ACCELERATE_CONFIG=configs/accelerate_config.yaml

accelerate launch \
  --config_file $ACCELERATE_CONFIG \
  src/training_scripts/pretrain_trainer.py \
  configs/pretrain_config.yaml
