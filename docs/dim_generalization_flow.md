# 维度泛化完整流程

## 流程概览

```
抽取阶段 → 符号化阶段 → 具体化阶段（LLM） → 泛化阶段 → 验证阶段
     ↓           ↓              ↓              ↓          ↓
  model.py   constraints    具体维度值      9个变体    run_model
  weight_meta
```

## 各阶段说明

### 1. 抽取阶段

**责任人**：抽取团队

**输出产物**：
- `model.py`：计算图代码（可能包含 torch.SymInt 参数）
- `weight_meta.py`：张量元数据

### 2. 符号化阶段（gen-constraints）

**责任人**：维度泛化 Agent

**核心思路**：分析所有 Tensor 的相似性，提取公共维度，符号化为 S0、S1

**示例**：
```
Tensor A: [1, 128, 768]
Tensor B: [1, 128, 1024]
Tensor C: [1, 64, 768]

分析相似性：
- 128 在 A、B 的 axis 1 都出现 → 公共维度，标记为 S0
- 768 在 A、C 的 axis 2 都出现 → 公共维度，标记为 S1

结果：
Tensor A: [1, S0, S1]
Tensor B: [1, S0, 1024]
Tensor C: [1, 64, S1]
```

**操作命令**：
```bash
python3.10 scripts/ops.py gen-constraints <subgraph_path>
```

**输出**：`input_tensor_constraints.py`（包含 sympy Symbol）

### 3. 具体化阶段（LLM Reify）

**责任人**：执行任务的 Agent

**核心思路**：针对符号模式（如 S0, S1），问 Agent 这些符号经常取哪些值

**操作步骤**：

```bash
# Step 1: 提取符号信息
python3.10 scripts/ops.py llm-reify <subgraph_path>

# 输出示例：
# 符号: S0, S1
# 符号化形状: [(1, S0, 768)]
# 当前示例值: {S0: 128}

# Step 2: Agent 根据信息推理具体维度值
# Agent 思考：对于 NLP 模型，S0 是 seq_len，常见值为 64, 128, 256, 512, 1024

# Step 3: 写入 Agent 推理的值
python3.10 scripts/ops.py llm-reify <subgraph_path> --values "S0=64,128,256,512,1024"
```

**输出**：`llm_reified_values.json`

### 4. 泛化阶段

**责任人**：维度泛化 Agent

**两种方式**：

```bash
# 方式一：使用预设 Reifier（默认）
python3.10 scripts/ops.py generalize <subgraph_path> \n    --output-dir <output_dir> --data-dir <data_dir>

# 方式二：使用 LLM 具体化的值
python3.10 scripts/ops.py generalize <subgraph_path> \n    --output-dir <output_dir> --data-dir <data_dir> --use-llm
```

### 5. 验证阶段

```bash
python3.10 scripts/ops.py verify <generalized_subgraph_path>
```

---

## 状态转换图

```
hardcoded (无SymInt，无约束)
    ↓ gen-constraints
needs_symbolize (无SymInt，有约束)
    ↓ symbolize
needs_constraints (有SymInt，无有效约束)
    ↓ gen-constraints
needs_reifier (有SymInt，有效约束，无Reifier)
    ↓ assign-reifier 或 llm-reify
ready_for_generalization (有SymInt，有效约束，有Reifier)
    ↓ generalize
完成
```

---

## 关键认知

### 符号化的本质

**分析 Tensor 相似性 → 提取公共维度 → 标记为 S0、S1**

这是给 AI 编译器支持动态维度时使用的标记方式。

### 具体化的本质

**问 Agent：某种符号模式经常取哪些值？**

- 符号模式 `(S0,)`：可能是 seq_len，常见值 64, 128, 256...
- 符号模式 `(S0, S1)`：可能是 (batch, seq_len)，常见组合 (1, 128), (16, 64)...

### 两种具体化方式

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| Reifier | 预设的维度值组合 | 通用场景，自动批量处理 |
| LLM Reify | Agent 推理具体值 | 需要模型专业知识，定制化场景 |

---

## 命令速查

```bash
# 符号化
python3.10 scripts/ops.py gen-constraints <sg_path>

# LLM 具体化
python3.10 scripts/ops.py llm-reify <sg_path>                    # 提取符号信息
python3.10 scripts/ops.py llm-reify <sg_path> --values "S0=..."  # 写入值

# Reifier 方式
python3.10 scripts/ops.py assign-reifier <sg_path>
python3.10 scripts/ops.py generalize <sg_path> --output-dir ... --data-dir ...

# LLM 方式
python3.10 scripts/ops.py generalize <sg_path> --output-dir ... --data-dir ... --use-llm
```