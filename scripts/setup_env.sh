#!/bin/bash
# DimGeneralizeAgent 环境设置
# 用法: source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh

export PYTHON=python3.10
export DIM_GEN_ROOT=/ssd1/liangtai-work/DimGeneralizeAgent
export GRAPHNET_ROOT=/ssd1/liangtai-work/GraphNet

# 五个待处理数据目录
export DATA_DIR_410=/ssd1/liangtai-work/lt_submit4.10
export DATA_DIR_47=/ssd1/liangtai-work/lt_submit4.7
export DATA_DIR_49=/ssd1/liangtai-work/lt_submit4.9
export DATA_DIR_413=/ssd1/liangtai-work/lt_submit4.13
export DATA_DIR_43=/ssd1/liangtai-work/lt_submit4.3

# 对应输出目录
export OUTPUT_DIR_410=/ssd1/liangtai-work/lt_submit4.10_dim_gen
export OUTPUT_DIR_47=/ssd1/liangtai-work/lt_submit4.7_dim_gen
export OUTPUT_DIR_49=/ssd1/liangtai-work/lt_submit4.9_dim_gen
export OUTPUT_DIR_413=/ssd1/liangtai-work/lt_submit4.13_dim_gen
export OUTPUT_DIR_43=/ssd1/liangtai-work/lt_submit4.3_dim_gen

echo "Environment set up (5 data dirs):"
echo "  4.10: $DATA_DIR_410 (~397 子图, 先跑通)"
echo "  4.7:  $DATA_DIR_47  (~484 子图)"
echo "  4.9:  $DATA_DIR_49  (~2067 子图)"
echo "  4.13: $DATA_DIR_413 (~2680 子图)"
echo "  4.3:  $DATA_DIR_43  (~3785 子图)"
