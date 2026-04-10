#!/bin/bash
# DimGeneralizeAgent 环境设置
# 用法: source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh

export PYTHON=python3.10
export DIM_GEN_ROOT=/ssd1/liangtai-work/DimGeneralizeAgent
export GRAPHNET_ROOT=/ssd1/liangtai-work/GraphNet

# 三个待处理数据目录（格式各不同，Agent 须先诊断再处理）
export DATA_DIR_43=/ssd1/liangtai-work/lt_submit4.3
export DATA_DIR_47=/ssd1/liangtai-work/lt_submit4.7
export DATA_DIR_49=/ssd1/liangtai-work/lt_submit4.9

# 对应输出目录
export OUTPUT_DIR_43=/ssd1/liangtai-work/lt_submit4.3_dim_gen
export OUTPUT_DIR_47=/ssd1/liangtai-work/lt_submit4.7_dim_gen
export OUTPUT_DIR_49=/ssd1/liangtai-work/lt_submit4.9_dim_gen

echo "Environment set up:"
echo "  PYTHON=$PYTHON"
echo "  DATA_DIR_43=$DATA_DIR_43  (lt_submit4.3 - 可能含硬编码维度)"
echo "  DATA_DIR_47=$DATA_DIR_47  (lt_submit4.7 - 已含 SymInt，缺约束)"
echo "  DATA_DIR_49=$DATA_DIR_49  (lt_submit4.9 - 格式待确认)"
