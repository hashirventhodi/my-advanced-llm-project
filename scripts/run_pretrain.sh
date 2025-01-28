#!/usr/bin/env bash

# Set PYTHONPATH to the project root dynamically
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Run accelerate
ACCELERATE_CONFIG=configs/accelerate_config.yaml
accelerate launch \
  --config_file $ACCELERATE_CONFIG \
  src/training_scripts/pretrain_trainer.py \
  configs/pretrain_config.yaml
