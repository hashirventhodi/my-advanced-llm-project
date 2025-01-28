#!/usr/bin/env bash

accelerate launch \
  src/training_scripts/reward_model_trainer.py \
  configs/reward_model_config.yaml
