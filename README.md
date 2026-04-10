# DimGeneralizeAgent

对 PyTorch FX 计算图执行维度泛化 —— 为每个子图生成 9 份不同维度参数的变体，用于提升计算图的泛化能力。

## 背景

在计算图提取场景中，每个子图（subgraph）的 `model.py` 包含硬编码的维度值或 `torch.SymInt` 符号维度。维度泛化的目标是将符号维度替换为 9 组不同的具体值，从而为下游任务提供多样化的计算图变体。

本项目依赖 [GraphNet](https://github.com/your-org/GraphNet) 框架的维度泛化能力，提供诊断工具和操作原语，支持 Agent 按需调度执行。

## 维度泛化 Pipeline

```
hardcoded ──(symbolize)──→ has SymInt
has SymInt ──(gen-constraints)──→ has constraints
has constraints ──(assign-reifier)──→ has reifier
has reifier ──(generalize)──→ 9 dimension variants
```

每步必须严格按顺序执行，后续步骤依赖前一步的产出。

### 9 组维度变体

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

## 快速开始

### 环境要求

- Python 3.10+（GraphNet 使用 `X | Y` 类型联合语法）
- GraphNet 框架

### 安装

```bash
# 1. 克隆本项目和 GraphNet
git clone https://github.com/your-org/DimGeneralizeAgent.git
git clone https://github.com/your-org/GraphNet.git

# 2. 安装自定义 Reifier 到 GraphNet
cp DimGeneralizeAgent/scripts/subgraph_sym_dim_reifier.py \
   GraphNet/graph_net/torch/sym_dim_reifiers/

# 3. 注册 Reifier（在 GraphNet/graph_net/torch/reifier_factory.py 的 get_reifier_names 中添加）
# "subgraph_sym_dim_reifier"
```

### 使用

```bash
# 设置环境
source scripts/setup_env.sh

# 1. 诊断：了解数据全貌
python3.10 scripts/diagnose.py /path/to/data --output /tmp/diag.json

# 2. 生成约束文件
python3.10 scripts/ops.py batch /path/to/data \
    --action gen-constraints --status-filter needs_constraints \
    --diag-json /tmp/diag.json

# 3. 分配 Reifier
python3.10 scripts/ops.py batch /path/to/data \
    --action assign-reifier --status-filter needs_reifier

# 4. 生成 9 份维度变体
python3.10 scripts/ops.py batch /path/to/data \
    --action generalize --status-filter ready_for_generalization \
    --output-dir /path/to/output

# 5. 更新进度
python3.10 scripts/ops.py snapshot
```

## 工具说明

### diagnose.py — 数据诊断

扫描计算图目录，输出每个子图的状态诊断：

| 状态 | 含义 | 下一步操作 |
|------|------|-----------|
| `hardcoded` | 无 SymInt，维度硬编码 | `symbolize` |
| `needs_constraints` | 有 SymInt，缺约束文件 | `gen-constraints` |
| `needs_reifier` | 有约束，缺 Reifier | `assign-reifier` |
| `ready_for_generalization` | 有 Reifier，可泛化 | `generalize` |
| `broken` | 缺失 model.py 或 forward | 跳过 |

```bash
python3.10 scripts/diagnose.py /path/to/data
python3.10 scripts/diagnose.py /path/to/data --output diag.json
python3.10 scripts/diagnose.py /path/to/data --status-filter hardcoded
python3.10 scripts/diagnose.py /path/to/data --model-name "bert-base"
```

### ops.py — 操作原语

提供单子图操作和批量操作：

```bash
# 单子图操作
python3.10 scripts/ops.py diagnose <subgraph_path>
python3.10 scripts/ops.py gen-constraints <subgraph_path>
python3.10 scripts/ops.py symbolize <subgraph_path>
python3.10 scripts/ops.py assign-reifier <subgraph_path>
python3.10 scripts/ops.py reify-preview <subgraph_path>
python3.10 scripts/ops.py generalize <subgraph_path> --output-dir <dir> --data-dir <dir>
python3.10 scripts/ops.py verify <subgraph_path>

# 批量操作
python3.10 scripts/ops.py batch <data_dir> --action <action> --status-filter <status>
python3.10 scripts/ops.py batch <data_dir> --action gen-constraints --diag-json /tmp/diag.json

# 更新进度文件
python3.10 scripts/ops.py snapshot
```

### subgraph_sym_dim_reifier.py — 自定义 Reifier

GraphNet 内置的 NLP/CV Reifier 只匹配整模型的形状模式（如 `[(1,S0,S1)]`）。子图级别的形状模式更碎片化（如 `[(1,S0,4096)]`、`[(S0,S1)]`），因此本项目提供了兜底 Reifier，匹配任何非空符号化形状。

## 输出结构

泛化后的计算图按维度变体编号组织：

```
output_dir/
├── 0/                              # 第 1 组维度
│   └── <model_name>/
│       └── <subgraph>/
│           ├── model.py
│           ├── input_tensor_constraints.py
│           ├── weight_meta.py
│           └── ...
├── 1/                              # 第 2 组维度
│   └── ...
└── 8/                              # 第 9 组维度
    └── ...
```

每组维度下保持与源数据相同的目录结构，仅约束文件中的维度值不同。

## 项目结构

```
DimGeneralizeAgent/
├── README.md
├── CLAUDE.md                          # Agent 入口（指向 PROGRESS.md）
├── PROGRESS.md                        # 进度跟踪（自动生成）
├── GUIDE.md                           # 详细工具文档
├── scripts/
│   ├── setup_env.sh                   # 环境变量设置
│   ├── diagnose.py                    # 数据诊断工具
│   ├── ops.py                         # 操作原语 + snapshot
│   └── subgraph_sym_dim_reifier.py    # 自定义 Reifier（需安装到 GraphNet）
├── config/
│   └── project.json                   # 项目配置与统计
└── .gitignore
```

## License

MIT
