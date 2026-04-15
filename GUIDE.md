# DimGeneralizeAgent - 计算图维度泛化 Agent

## 项目目标

对五个数据目录下的所有计算图执行维度泛化，为每个子图生成 9 份不同维度版本。

## 核心原则: 先诊断，再决策，后执行

**不要盲目执行脚本！** 每个数据目录的格式不同，每个子图的状态不同。Agent 必须：
1. 先运行 `diagnose.py` 了解全貌
2. 根据诊断结果分析每个子图需要什么操作
3. 选择正确的操作原语组合执行

## 数据概况

五个数据目录，格式各不相同：

| 目录 | 子图数 | 特点 |
|------|--------|------|
| lt_submit4.10 | 397 | 从GraphNet_20260403随机抽100图，98%硬编码 |
| lt_submit4.7 | 484 | 86%已有SymInt，最小规模 |
| lt_submit4.9 | 2,064 | 63%需约束，37%硬编码 |
| lt_submit4.13 | 2,687 | 82%硬编码 |
| lt_submit4.3 | 3,777 | 52%需约束，48%硬编码 |

**重要**: 必须使用 `python3.10` 运行（GraphNet 使用了 Python 3.10 的 `X | Y` 类型联合语法）。

## 关键概念区分

### 符号化 vs SymInt 参数化（重要！）

| 概念 | 阶段 | 说明 |
|------|------|------|
| **符号化** | gen-constraints | 分析 Tensor shapes，找公共维度，用 sympy Symbol 标记，生成 `input_tensor_constraints.py` |
| **SymInt 参数化** | symbolize | 基于约束文件，给 model.py 添加 torch.SymInt 参数 |

**两者无关！** 符号化用的是 sympy Symbol，SymInt 是 PyTorch 的符号整数类型。

## 子图状态与决策树

诊断工具为每个子图标记以下状态之一，Agent 根据状态决定操作：

```
子图状态:
├── broken              → 跳过（无 model.py 或无 forward 方法）
├── hardcoded           → 执行 gen-constraints（生成约束文件）
├── needs_symbolize     → 执行 symbolize（SymInt 参数化）
├── needs_constraints   → 执行 gen-constraints（生成约束文件）
├── needs_reifier       → 执行 assign-reifier（分配 Reifier）
└── ready_for_generalization → 执行 generalize（生成 9 份副本）
```

**处理顺序**: `gen-constraints → symbolize → assign-reifier → generalize`

**注意**: `gen-constraints` 对所有子图都适用（有 SymInt 和无 SymInt）。`symbolize` 仅用于硬编码子图（无 SymInt）。

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

# 生成约束文件（符号化，适用于所有子图）
$PYTHON $DIM_GEN_ROOT/scripts/ops.py gen-constraints <sg_path>

# SymInt 参数化（仅用于硬编码子图，需先执行 gen-constraints）
$PYTHON $DIM_GEN_ROOT/scripts/ops.py symbolize <sg_path>

# 分配 Reifier（针对 needs_reifier 状态）
$PYTHON $DIM_GEN_ROOT/scripts/ops.py assign-reifier <sg_path>

# 预览 Reifier 的 9 组维度值
$PYTHON $DIM_GEN_ROOT/scripts/ops.py reify-preview <sg_path>

# 生成 9 份维度泛化副本
$PYTHON $DIM_GEN_ROOT/scripts/ops.py generalize <sg_path> \n    --output-dir <output_dir> --data-dir <data_dir>

# 验证子图是否可运行
$PYTHON $DIM_GEN_ROOT/scripts/ops.py verify <sg_path>

# 批量操作（慎用，建议先诊断确认范围）
$PYTHON $DIM_GEN_ROOT/scripts/ops.py batch <data_dir> \n    --action gen-constraints --status-filter hardcoded
$PYTHON $DIM_GEN_ROOT/scripts/ops.py batch <data_dir> \n    --action gen-constraints --status-filter needs_constraints
$PYTHON $DIM_GEN_ROOT/scripts/ops.py batch <data_dir> \n    --action symbolize --status-filter needs_symbolize
$PYTHON $DIM_GEN_ROOT/scripts/ops.py batch <data_dir> \n    --action assign-reifier --status-filter needs_reifier
$PYTHON $DIM_GEN_ROOT/scripts/ops.py batch <data_dir> \n    --action generalize --status-filter ready_for_generalization \n    --output-dir <output_dir>
```

## Agent 工作流

### Step 1: 环境设置

```bash
source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh
```

### Step 2: 诊断各目录

对每个数据目录运行 diagnose.py，了解：
- 状态分布（hardcoded / needs_symbolize / needs_constraints / needs_reifier / ready_for_generalization / broken）
- SymInt 参数数量分布
- 约束文件状态
- Reifier 分配情况
- 主要形状模式

### Step 3: 按状态逐批处理

对每个数据目录，按状态从低到高处理：

1. **hardcoded 子图** → `ops.py gen-constraints`（生成约束文件）
   - 分析 tensor shapes，找公共维度
   - 生成 `input_tensor_constraints.py`

2. **needs_symbolize 子图** → `ops.py symbolize`（SymInt 参数化）
   - 仅用于硬编码子图（gen-constraints 后的状态）
   - 使用 GraphNet 的 `DimensionSymbolizer`
   - 可能对部分模型失败，失败的记录下来

3. **needs_constraints 子图** → `ops.py gen-constraints`（生成约束文件）
   - 已有 SymInt 参数
   - 分析 SymInt example_value 识别动态维度
   - 生成 `input_tensor_constraints.py`

4. **needs_reifier 子图** → `ops.py assign-reifier`（分配 Reifier）
   - Reifier 匹配顺序: naive_cv → naive_nlp → subgraph（兜底）
   - subgraph_sym_dim_reifier 匹配任何非空符号化形状，确保不会漏掉

5. **ready_for_generalization 子图** → `ops.py generalize`（生成 9 份副本）

### Step 4: 验证

**验证泛化后的子图**（不是原始子图）：

```bash
# 从泛化输出目录随机抽样验证
$PYTHON $DIM_GEN_ROOT/scripts/ops.py verify /ssd1/liangtai-work/lt_submit4.7_dim_gen/0/<model_name>/<subgraph>
```

注意：原始子图的 verify 可能失败（多个 SymInt 参数未全部推断），但泛化后的子图应该能通过。

### Step 5: 汇报统计

完成所有步骤后，汇报：
- 各状态处理成功/跳过/失败数量
- 硬编码子图数量
- 损坏子图数量

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
├── CLAUDE.md                          # Agent 入口（指向 PROGRESS.md）
├── PROGRESS.md                        # 当前进度和任务
├── GUIDE.md                           # 本文件（详细工具文档）
├── scripts/
│   ├── setup_env.sh                   # 环境设置
│   ├── diagnose.py                    # 数据诊断工具
│   ├── ops.py                         # 操作原语
│   └── subgraph_sym_dim_reifier.py    # 自定义 Reifier
├── docs/
│   ├── dim_generalization_flow.md     # 维度泛化流程文档
│   ├── extraction_quality_requirements.md  # 抽取质量要求
│   ├── fix_symint_example_value.md    # SymInt 修复指南
│   └── fix_dim_gen_error.md           # 泛化后修复指南
└── logs/                              # 运行日志（自动生成）
```

## 关键依赖路径

| 项目 | 路径 |
|------|------|
| 数据目录 4.10 | `/ssd1/liangtai-work/lt_submit4.10/` |
| 数据目录 4.7 | `/ssd1/liangtai-work/lt_submit4.7/` |
| 数据目录 4.9 | `/ssd1/liangtai-work/lt_submit4.9/` |
| 数据目录 4.13 | `/ssd1/liangtai-work/lt_submit4.13/` |
| 数据目录 4.3 | `/ssd1/liangtai-work/lt_submit4.3/` |
| 输出 4.10 | `/ssd1/liangtai-work/lt_submit4.10_dim_gen/` |
| 输出 4.7 | `/ssd1/liangtai-work/lt_submit4.7_dim_gen/` |
| 输出 4.9 | `/ssd1/liangtai-work/lt_submit4.9_dim_gen/` |
| 输出 4.13 | `/ssd1/liangtai-work/lt_submit4.13_dim_gen/` |
| 输出 4.3 | `/ssd1/liangtai-work/lt_submit4.3_dim_gen/` |
| GraphNet 框架 | `/ssd1/liangtai-work/GraphNet/` |

## Agent 调度 Prompt

新开 Agent Session 时，直接粘贴以下 prompt 即可开始工作。

### 极简版（推荐）

```
source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh
读 /ssd1/liangtai-work/DimGeneralizeAgent/PROGRESS.md，按"当前任务"执行对应操作。
每步完成后运行 `python3.10 scripts/ops.py snapshot` 更新进度。
```

### 指定目录版

处理特定目录时，替换 `<目录名>` 为 `4.10`、`4.3`、`4.7`、`4.9` 或 `4.13`：

```
source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh
对 lt_submit<目录名> 执行维度泛化。
1. diagnose.py 诊断 → 2. batch gen-constraints → 3. batch symbolize → 4. batch assign-reifier → 5. batch generalize → 6. verify 抽样验证
每步完成后运行 `python3.10 scripts/ops.py snapshot` 更新进度。
```

## 技术要点

1. **必须用 python3.10**: GraphNet 使用了 Python 3.10 的 `X | Y` 类型语法
2. **gen-constraints 对所有子图适用**: 无论是否有 SymInt，都会分析 shapes 生成约束
3. **symbolize 仅用于硬编码子图**: 给没有 SymInt 的 model.py 添加 SymInt 参数
4. **符号化顺序**: 优先用 SymInt example_value，其次用高频维度（出现次数 >= 2）
5. **自定义 Reifier 兜底**: 匹配任何非空符号化形状，确保不会漏掉
6. **批量操作用 batch 子命令**: 支持 `--status-filter` 按状态过滤