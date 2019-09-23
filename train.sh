#!/bin/bash
source ../../horovod_env.sh
python train.py \
--opt_dir options/UnetPlus6.json #指定配置文件路径