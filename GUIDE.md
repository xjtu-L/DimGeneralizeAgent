# DimGeneralizeAgent - 计算图维度泛化 Agent

## 项目目标

对三个数据目录下的所有计算图执行维度泛化，为每个子图生成 9 份不同维度版本。

## 核心原则: 先诊断，再决策，后执行

**不要盲目执行脚本！** 每个数据目录的格式不同，每个子图的状态不同。Agent 必须：
1. 先运行 `diagnose.py` 了解全貌
2. 根据诊断结果分析每个子图需要什么操作
3. 选择正确的操作原语组合执行

## 数据概况

三个数据目录，格式各不相同：

| 目录 | 子图数 | 需约束 | 硬编码 | 损坏 | 特点 |
|------|--------|--------|--------|------|------|
| lt_submit4.3 | 3,777 | 1,947 | 1,809 | 21 | 最多硬编码(48%)，需 FX Pass 符号化 |
| lt_submit4.7 | 484 | 415 | 67 | 2 | 大部分已含 SymInt(86%)，最小规模 |
| lt_submit4.9 | 2,064 | 1,304 | 755 | 5 | 介于两者之间 |
| **合计** | **6,325** | **3,666** | **2,631** | **28** | |

**共同特点**: model.py 已包含 `torch.SymInt`(s0, s1...)，但 `input_tensor_constraints.py` 几乎全部为空。少数子图完全没有 SymInt（硬编码维度）。

**重要**: 必须使用 `python3.10` 运行（GraphNet 使用了 Python 3.10 的 `X | Y` 类型联合语法）。

## 子图状态与决策树

诊断工具为每个子图标记以下状态之一，Agent 根据状态决定操作：

```
子图状态:
├── broken          → 跳过（无 model.py 或无 forward 方法）
├── hardcoded       → 需先符号化 (op: symbolize)，再走 needs_constraints 流程
├── needs_constraints → 已有 SymInt，生成约束文件 (op: gen-constraints)
├── needs_reifier   → 已有约束，分配 Reifier (op: assign-reifier)
└── ready_for_generalization → 已有 Reifier，直接泛化 (op: generalize)
```

**处理顺序必须严格**: `symbolize → gen-constraints → assign-reifier → generalize`

## 工具使用

### 1. 诊断工具: diagnose.py — 先用这个了解数据

```bash
source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh

# 全局诊断（查看状态分布）
$PYTHON $DIM_GEN_ROOT/scripts/diagnose.py /ssd1/liangtai-work/lt_submit4.7

# 保存详细结果到 JSON
$PYTHON $DIM_GEN_ROOT/scripts/diagnose.py /ssd1/liangtai-work/lt_submit4.7 --output /tmp/diag.json

# 过滤特定状态的子图
$PYTHON $DIM_GEN_ROOT/scripts/diagnose.py /ssd1/liangtai-work/lt_submit4.3 --status-filter hardcoded

# 只看某个模型
$PYTHON $DIM_GEN_ROOT/scripts/diagnose.py /ssd1/liangtai-work/lt_submit4.7 --model-name "01-ai_Yi-1.5-6B-Chat"
```

### 2. 操作原语: ops.py — 根据诊断结果选择操作

```bash
# 诊断单个子图（查看详细状态）
$PYTHON $DIM_GEN_ROOT/scripts/ops.py diagnose <sg_path>

# 生成约束文件（针对 needs_constraints 状态）
$PYTHON $DIM_GEN_ROOT/scripts/ops.py gen-constraints <sg_path>

# FX Pass 符号化（针对 hardcoded 状态）
$PYTHON $DIM_GEN_ROOT/scripts/ops.py symbolize <sg_path>

# 分配 Reifier（针对 needs_reifier 状态）
$PYTHON $DIM_GEN_ROOT/scripts/ops.py assign-reifier <sg_path>

# 预览 Reifier 的 9 组维度值
$PYTHON $DIM_GEN_ROOT/scripts/ops.py reify-preview <sg_path>

# 生成 9 份维度泛化副本
$PYTHON $DIM_GEN_ROOT/scripts/ops.py generalize <sg_path> \
    --output-dir <output_dir> --data-dir <data_dir>

# 验证子图是否可运行
$PYTHON $DIM_GEN_ROOT/scripts/ops.py verify <sg_path>

# 批量操作（慎用，建议先诊断确认范围）
$PYTHON $DIM_GEN_ROOT/scripts/ops.py batch <data_dir> \
    --action gen-constraints --status-filter needs_constraints
$PYTHON $DIM_GEN_ROOT/scripts/ops.py batch <data_dir> \
    --action assign-reifier --status-filter needs_reifier
$PYTHON $DIM_GEN_ROOT/scripts/ops.py batch <data_dir> \
    --action generalize --status-filter ready_for_generalization \
    --output-dir <output_dir>
```

## Agent 工作流

### Step 1: 环境设置

```bash
source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh
```

### Step 2: 诊断各目录

对每个数据目录运行 diagnose.py，了解：
- 状态分布（hardcoded / needs_constraints / needs_reifier / ready_for_generalization / broken）
- SymInt 参数数量分布
- 约束文件状态
- Reifier 分配情况
- 主要形状模式

### Step 3: 按状态逐批处理

对每个数据目录，按状态从低到高处理：

1. **hardcoded 子图** → `ops.py symbolize`（FX Pass 符号化）
   - 注意: 此步骤依赖 GraphNet 的 `DimensionSymbolizer`，可能对部分模型失败
   - 失败的记录下来，后续可手动处理或跳过

2. **needs_constraints 子图** → `ops.py gen-constraints`（生成约束文件）
   - 从 forward 签名提取 SymInt 参数
   - 从 weight_meta.py 获取第一个输入 tensor 的形状
   - 使用 `DynamicDimConstraints.make_by_named_inputs()` + `symbolize()` API
   - 符号化顺序: axis 1 (seq_len) → axis 0 (batch) → 兜底任意 >1 维度

3. **needs_reifier 子图** → `ops.py assign-reifier`（分配 Reifier）
   - Reifier 匹配顺序: naive_cv → naive_nlp → subgraph（兜底）
   - subgraph_sym_dim_reifier 匹配任何非空符号化形状，确保不会漏掉

4. **ready_for_generalization 子图** → `ops.py generalize`（生成 9 份副本）

### Step 4: 验证

随机抽样验证泛化结果：
```bash
$PYTHON $DIM_GEN_ROOT/scripts/ops.py verify <output_dir>/0/<model_name>/<subgraph>
```

## 自定义 Reifier: subgraph_sym_dim_reifier

由于子图级别的输入形状模式（如 `[(1,S0,4096)]`、`[(S0,S1)]`）与 GraphNet 内置的 NLP/CV Reifier 不匹配，已安装自定义 Reifier 作为兜底。

**9 组维度值**:

| 编号 | 单符号 (seq_len) | 双符号 (batch, seq_len) |
|------|-----------------|----------------------|
| 0 | 64 | 1, 64 |
| 1 | 512 | 1, 512 |
| 2 | 128 | 16, 128 |
| 3 | 64 | 32, 64 |
| 4 | 256 | 8, 256 |
| 5 | 512 | 4, 512 |
| 6 | 1024 | 2, 1024 |
| 7 | 128 | 64, 128 |
| 8 | 64 | 128, 64 |

## 目录结构

```
DimGeneralizeAgent/
├── CLAUDE.md                          # 本文件（Agent 入口指南）
├── scripts/
│   ├── setup_env.sh                   # 环境设置（含 PYTHON=python3.10, 三个数据目录变量）
│   ├── diagnose.py                    # 数据诊断工具
│   ├── ops.py                         # 操作原语（gen-constraints, symbolize, assign-reifier, etc.）
│   └── subgraph_sym_dim_reifier.py    # 自定义 Reifier（已安装到 GraphNet）
├── config/
│   └── project.json                   # 三个数据目录的统计信息
└── logs/                              # 运行日志（自动生成）
```

## 关键依赖路径

| 项目 | 路径 |
|------|------|
| 数据目录 4.3 | `/ssd1/liangtai-work/lt_submit4.3/` |
| 数据目录 4.7 | `/ssd1/liangtai-work/lt_submit4.7/` |
| 数据目录 4.9 | `/ssd1/liangtai-work/lt_submit4.9/` |
| 输出 4.3 | `/ssd1/liangtai-work/lt_submit4.3_dim_gen/` |
| 输出 4.7 | `/ssd1/liangtai-work/lt_submit4.7_dim_gen/` |
| 输出 4.9 | `/ssd1/liangtai-work/lt_submit4.9_dim_gen/` |
| GraphNet 框架 | `/ssd1/liangtai-work/GraphNet/` |
| 自定义 Reifier (源码) | `scripts/subgraph_sym_dim_reifier.py` |
| 自定义 Reifier (安装) | `GraphNet/graph_net/torch/sym_dim_reifiers/subgraph_sym_dim_reifier.py` |
| Reifier Factory | `GraphNet/graph_net/torch/reifier_factory.py` |

## Agent 调度 Prompt

新开 Agent Session 时，直接粘贴以下 prompt 即可开始工作。

### 单目录版（推荐，每次处理一个目录）

```
source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh

请对 /ssd1/liangtai-work/lt_submit4.7 执行维度泛化任务。

工作流程：
1. 运行 diagnose.py 诊断，保存结果到 JSON：
   python3.10 /ssd1/liangtai-work/DimGeneralizeAgent/scripts/diagnose.py /ssd1/liangtai-work/lt_submit4.7 --output /tmp/diag_47.json

2. 读取诊断结果，分析状态分布

3. 按 status 逐批处理（用 batch --diag-json 避免重复扫描）：
   a. needs_constraints → batch gen-constraints --status-filter needs_constraints --diag-json /tmp/diag_47.json
   b. 重新诊断保存到新 JSON，needs_reifier → batch assign-reifier --status-filter needs_reifier --diag-json <新JSON>
   c. 重新诊断保存到新 JSON，ready_for_generalization → batch generalize --status-filter ready_for_generalization --output-dir /ssd1/liangtai-work/lt_submit4.7_dim_gen --diag-json <新JSON>

4. 硬编码子图（hardcoded）单独分析，看能不能 symbolize

5. 汇报最终统计

注意：每步完成后重新诊断，确保状态正确流转。遇到大量错误要停下来分析原因。
```

### 全量版（三个目录一次完成）

```
source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh

请依次对三个数据目录执行维度泛化任务。按规模从小到大处理：

目录顺序：
1. /ssd1/liangtai-work/lt_submit4.7 (484 子图，先跑通)
2. /ssd1/liangtai-work/lt_submit4.9 (2,064 子图)
3. /ssd1/liangtai-work/lt_submit4.3 (3,777 子图)

每个目录的工作流程：
1. diagnose.py 诊断，保存 JSON（如 /tmp/diag_47.json）
2. batch gen-constraints --status-filter needs_constraints --diag-json /tmp/diag_47.json
3. 重新诊断 → batch assign-reifier --status-filter needs_reifier --diag-json <新JSON>
4. 重新诊断 → batch generalize --status-filter ready_for_generalization --output-dir <对应output_dir> --diag-json <新JSON>
5. 分析 hardcoded 子图

每个目录完成后汇报统计，再进入下一个。
```

### 换目录只需改两处

| 变量 | lt_submit4.3 | lt_submit4.7 | lt_submit4.9 |
|------|-------------|-------------|-------------|
| 数据目录 | `/ssd1/liangtai-work/lt_submit4.3` | `/ssd1/liangtai-work/lt_submit4.7` | `/ssd1/liangtai-work/lt_submit4.9` |
| 输出目录 | `/ssd1/liangtai-work/lt_submit4.3_dim_gen` | `/ssd1/liangtai-work/lt_submit4.7_dim_gen` | `/ssd1/liangtai-work/lt_submit4.9_dim_gen` |
| 诊断JSON | `/tmp/diag_43.json` | `/tmp/diag_47.json` | `/tmp/diag_49.json` |

## 技术要点

1. **必须用 python3.10**: GraphNet 使用了 Python 3.10 的 `X | Y` 类型语法
2. **三个目录格式相似但比例不同**: 都有 SymInt 和硬编码子图，但硬编码比例差异大
3. **约束生成只取第一个输入 tensor**: 避免多个 tensor 在同一 axis 有不同维度导致 `symbolize()` 断言失败
4. **符号化顺序**: axis 1 (seq_len) → axis 0 (batch) → 兜底任意 >1
5. **自定义 Reifier 兜底**: 匹配任何非空符号化形状，确保不会漏掉
6. **批量操作用 batch 子命令**: 支持 `--status-filter` 按状态过滤，支持 `--resume` 断点续传
7. **磁盘空间**: 6,325 子图 × 9 = 56,925 个输出目录
