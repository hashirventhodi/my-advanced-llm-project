#!/usr/bin/env bash

accelerate launch \
  src/training_scripts/sft_trainer.py \
  configs/sft_config.yaml
