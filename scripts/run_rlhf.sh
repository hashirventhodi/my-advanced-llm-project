#!/usr/bin/env bash

# RLHF script placeholder
accelerate launch \
  src/training_scripts/rlhf_trainer.py \
  configs/rlhf_config.yaml
