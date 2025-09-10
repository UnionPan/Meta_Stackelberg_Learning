#!/bin/bash

# This script runs the RPS training with minimal settings to test the implementation.

export PYTHONPATH=$PYTHONPATH:$(pwd)

python train_rps.py \
    --num_training_iterations=100 \
    --rollout_length=256
