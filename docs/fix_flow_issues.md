# 维度泛化流程修复记录

**日期**: 2026-04-15

## 发现的问题

### 问题一：gen-constraints 依赖 SymInt

**现象**：`gen-constraints` 操作会跳过没有 SymInt 参数的子图。

**原因**：原逻辑用 SymInt example_value 识别动态维度，没有 SymInt 就无法工作。

**影响**：hardcoded 子图（无 SymInt）无法生成约束文件，形成死锁。

### 问题二：symbolize 命名和逻辑错误

**现象**：
- 操作名为"FX Pass 符号化"，与"符号化"概念混淆
- 没有检查约束文件是否存在就调用 DimensionSymbolizer

**原因**：
- DimensionSymbolizer 需要约束文件存在才能工作
- 但 hardcoded 子图没有约束文件

### 问题三：状态判断逻辑不匹配

**现象**：诊断状态与处理流程不一致。

**原因**：没有 `needs_symbolize` 状态来区分"有约束但无 SymInt"的子图。

---

## 修复方案

### 1. gen-constraints：独立分析 shapes

**修复文件**: `scripts/ops.py`

**修复内容**:
```python
def op_gen_constraints(sg_path):
    """生成 input_tensor_constraints.py（符号化）

    符号化：分析 tensor shapes，找公共维度，用 sympy Symbol 标记。
    与 SymInt 无关，适用于有 SymInt 和无 SymInt 的子图。
    """
    # 确定哪些维度是动态的
    # 策略：分析动态输入 tensor 的 shapes，找出公共维度（高频维度）

    # 如果有 SymInt，用 SymInt example_value 识别动态维度
    symint_values = set()
    if symint_params:
        for t in tensors:
            if t["name"] in symint_params and t.get("data") is not None:
                for v in (t["data"] if isinstance(t["data"], list) else [t["data"]]):
                    if isinstance(v, int) and v > 1:
                        symint_values.add(v)

    # 分析动态输入的维度频率
    dim_freq = Counter()  # 维度值 -> 出现次数
    for idx, (shape, name) in enumerate(dyn_dim_cstrs.input_shapes):
        if name not in dynamic_set:
            continue
        for axis, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 1:
                dim_freq[dim] += 1

    # 确定 candidate 维度值
    candidate_dims = set()
    if symint_values:
        candidate_dims = symint_values
    else:
        # 找出高频维度（出现次数 >= 2）
        for dim, count in dim_freq.items():
            if count >= 2:
                candidate_dims.add(dim)
```

### 2. symbolize：正确处理 SymInt 参数化

**修复文件**: `scripts/ops.py`

**修复内容**:
```python
def op_symbolize(sg_path):
    """SymInt 参数化：基于约束文件，给 model.py 添加 SymInt 参数

    仅用于 hardcoded 子图（没有 SymInt 参数的子图）。
    需要先运行 gen-constraints 生成约束文件。
    """
    # 检查约束文件是否存在
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    if not os.path.exists(cstr_path):
        print(f"SKIP: No constraints file in {sg_path} (run gen-constraints first)")
        return False
```

### 3. diagnose.py：调整状态判断逻辑

**修复文件**: `scripts/diagnose.py`

**修复内容**:
```python
# 5. Determine status
# 新流程:
# 1. gen-constraints (所有子图) -> 生成约束文件
# 2. symbolize (仅 hardcoded) -> 添加 SymInt
# 3. assign-reifier -> generalize

if not symint_params:
    # 没有 SymInt 的子图
    if r["constraint_status"] == "valid":
        # 已有约束文件，可以执行 symbolize
        r["status"] = "needs_symbolize"
    else:
        # 需要先执行 gen-constraints
        r["status"] = "hardcoded"
elif r["constraint_status"] == "valid":
    # 有 SymInt 且有有效约束
    if r["reifier"]:
        r["status"] = "ready_for_generalization"
    else:
        r["status"] = "needs_reifier"
else:
    # 有 SymInt 但没有有效约束
    r["status"] = "needs_constraints"
```

---

## 修复后的正确流程

### 状态转换图

```
hardcoded (无SymInt，无约束)
    ↓ gen-constraints
needs_symbolize (无SymInt，有约束)
    ↓ symbolize
needs_constraints (有SymInt，无有效约束)
    ↓ gen-constraints
needs_reifier (有SymInt，有效约束，无Reifier)
    ↓ assign-reifier
ready_for_generalization (有SymInt，有效约束，有Reifier)
    ↓ generalize
完成
```

### 处理流程

```bash
# 1. 所有子图执行 gen-constraints（符号化）
python3.10 scripts/ops.py batch <data_dir> --action gen-constraints --status-filter hardcoded
python3.10 scripts/ops.py batch <data_dir> --action gen-constraints --status-filter needs_constraints

# 2. 硬编码子图执行 symbolize（SymInt 参数化）
python3.10 scripts/ops.py batch <data_dir> --action symbolize --status-filter needs_symbolize

# 3. 后续流程
python3.10 scripts/ops.py batch <data_dir> --action assign-reifier --status-filter needs_reifier
python3.10 scripts/ops.py batch <data_dir> --action generalize --status-filter ready_for_generalization --output-dir <output_dir>
```

---

## 关键认知修正

### 符号化 vs SymInt 参数化

| 概念 | 类型 | 操作 | 作用 | 文件 |
|------|------|------|------|------|
| sympy Symbol (S0, S1) | sympy 符号 | gen-constraints | 标记动态维度 | input_tensor_constraints.py |
| torch.SymInt (s0, s1) | PyTorch 类型 | symbolize | 模型参数类型 | model.py |

**两者无关！**

- **符号化**：分析 Tensor shapes，找公共维度，用 sympy Symbol 标记
- **SymInt 参数化**：基于约束文件，给 model.py 添加 torch.SymInt 参数

### 问题分类

| 问题类型 | 出现阶段 | 责任人 | 参考文档 |
|---------|---------|--------|---------|
| SymInt example_value 错误 | 抽取阶段 | 抽取团队 | extraction_quality_requirements.md |
| 缺少约束文件 | 符号化阶段 | 泛化 Agent | dim_generalization_flow.md |
| reshape/view 错误 | 泛化后 | 泛化 Agent | fix_dim_gen_error.md |
| pos_embed 不匹配 | 泛化后 | 泛化 Agent | fix_dim_gen_error.md |

---

## 修改的文件清单

| 文件 | 修改内容 |
|------|---------|
| `scripts/ops.py` | op_gen_constraints: 独立分析 shapes，不依赖 SymInt |
| `scripts/ops.py` | op_symbolize: 检查约束文件，明确是 SymInt 参数化 |
| `scripts/ops.py` | op_snapshot: 更新状态统计和 PROGRESS.md 模板 |
| `scripts/diagnose.py` | diagnose_subgraph: 新增 needs_symbolize 状态 |
| `GUIDE.md` | 更新流程说明，明确区分符号化和 SymInt 参数化 |
| `docs/dim_generalization_flow.md` | 更新流程文档 |
| `MEMORY.md` | 更新技术要点和关键认知 |