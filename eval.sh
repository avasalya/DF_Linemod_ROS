#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

rosrun densefusion_ros densefusion_ros.py --model=txonigiri