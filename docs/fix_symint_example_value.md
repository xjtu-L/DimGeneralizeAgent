# 模型代码修复指南

## 问题一：SymInt Example Value 错误

### 问题背景

计算图的 `model.py` 里包含 SymInt 参数（如 `s0`, `L_self_head_dim`），但 `weight_meta.py` 里的 `example_value` 在上游抽取时未正确推断，都是 placeholder 值 4。

这导致 `run_model` 执行 forward 时维度不匹配。

## 问题示例：subgraph_1

### model.py forward 签名

```python
def forward(self, s0 : torch.SymInt, L_hidden_states_ : torch.Tensor,
            ..., L_self_head_dim : torch.SymInt, ..., s3 : torch.SymInt,
            L_position_embeddings_0_ : torch.Tensor, ...):
    view = linear.view((1, s0, -1, l_self_head_dim))
    mul = query_states * cos  # cos 来自 position_embeddings
```

### weight_meta.py（修复前）

```python
class Program_weight_tensor_meta_s0:
    name = "s0"
    shape = []
    data = [4]  # placeholder，错误

class Program_weight_tensor_meta_L_self_head_dim:
    name = "L_self_head_dim"
    shape = []
    data = [4]  # placeholder，错误

class Program_weight_tensor_meta_s3:
    name = "s3"
    shape = []
    data = [4]  # placeholder
```

### 关键 Tensor Shapes

```python
L_hidden_states_: [1, 32, 3584]      # s0 应该是 32
L_position_embeddings_0_: [1, 32, 128]  # L_self_head_dim 应该是 128
```

### 运行错误

```
RuntimeError: The size of tensor a (4) must match the size of tensor b (128)
```

原因：`L_self_head_dim=4`，但 `position_embeddings` 最后一维是 128。

## 修复方法

### 推断规则

1. **s0**：从输入 tensor（非权重）的第二维推断，通常是 seq_len
2. **L_self_head_dim**：从 `position_embeddings` 的最后一维推断
3. **其他 sX 参数**：从高频维度推断（出现次数多的维度）

### 修复脚本

```python
import re
from collections import Counter

def fix_symint_values(sg_path):
    """从 tensor shapes 推断并修复 SymInt 的 example_value"""

    wm_path = f"{sg_path}/weight_meta.py"
    with open(wm_path) as f:
        content = f.read()

    # 1. 提取所有 tensor 的 name 和 shape
    tensors = []
    for match in re.finditer(r'class\s+\w+:\s*\n((?:\s+\w+\s*=\s*.*\n)*)', content):
        name = shape = None
        for line in match.group(1).strip().split('\n'):
            line = line.strip()
            if line.startswith('name'):
                name = line.split('=', 1)[1].strip().strip('"')
            elif line.startswith('shape'):
                shape = eval(line.split('=', 1)[1].strip())
        if name and shape is not None:
            tensors.append({"name": name, "shape": shape})

    # 2. 解析 SymInt 参数
    model_path = f"{sg_path}/model.py"
    with open(model_path) as f:
        model_content = f.read()
    m = re.search(r'def forward\(self,\s*(.*?)\):', model_content, re.DOTALL)
    symint_params = re.findall(r'(\w+)\s*:\s*torch\.SymInt', m.group(1))

    if not symint_params:
        return {}

    # 3. 统计维度频率
    dim_counter = Counter()
    for t in tensors:
        if t["shape"]:
            for dim in t["shape"]:
                if isinstance(dim, int) and dim > 1:
                    dim_counter[dim] += 1

    sorted_dims = sorted(dim_counter.items(), key=lambda x: -x[1])

    # 4. 推断 SymInt 值
    symint_value = {}

    # s0：输入 tensor 的第二维
    for t in tensors:
        if t["name"].startswith("L_") and "_parameters_" not in t["name"] and t["shape"]:
            if len(t["shape"]) >= 2 and t["shape"][1] > 1:
                symint_value["s0"] = t["shape"][1]
                break

    # L_self_head_dim：position_embeddings 的最后一维
    for t in tensors:
        if "position_embeddings" in t["name"] and t["shape"]:
            last_dim = t["shape"][-1]
            if last_dim > 1:
                symint_value["L_self_head_dim"] = last_dim
                break

    # 其他 sX 参数：从高频维度推断
    other_syms = [s for s in symint_params if s not in symint_value]
    used_values = set(symint_value.values())
    for sym in other_syms:
        for dim, count in sorted_dims:
            if dim not in used_values:
                symint_value[sym] = dim
                used_values.add(dim)
                break

    # 5. 更新 weight_meta.py
    for sym, val in symint_value.items():
        pattern = rf'(class\s+\w+:\s*\n(?:\s+\w+\s*=\s*.*\n)*?\s+name\s*=\s*"{sym}"\s*\n(?:\s+\w+\s*=\s*.*\n)*?)\s+data\s*=\s*\[.*?\]'
        replacement = rf'\1\n\tdata = [{val}]'
        content = re.sub(pattern, replacement, content)

    with open(wm_path, "w") as f:
        f.write(content)

    return symint_value
```

### 使用方法

```bash
# 修复单个子图
python3.10 -c "
from fix_symint import fix_symint_values
result = fix_symint_values('/ssd1/liangtai-work/lt_submit4.10_sample_10/subgraph_1')
print(result)
"

# 批量修复
for sg in /ssd1/liangtai-work/lt_submit4.10_sample_10/subgraph_*; do
    python3.10 -c "from fix_symint import fix_symint_values; print(fix_symint_values('$sg'))"
done
```

## 修复后验证

```bash
python3.10 /ssd1/liangtai-work/DimGeneralizeAgent/scripts/ops.py verify /ssd1/liangtai-work/lt_submit4.10_sample_10/subgraph_1
# 输出: OK: runs successfully
```

## 注意事项

1. 此修复应在 `gen-constraints` 之前执行，否则约束生成时 example_value 仍是错误的
2. 某些 SymInt 参数（如 `L_self_modules_mlp_modules_c_fc_nf`）需要从对应 weight 矩阵的 shape 推断，通用脚本可能无法覆盖
3. 修复后建议重新运行 `diagnose.py` 确认状态

---

## 问题二：_assert_scalar API 兼容性问题

### 问题背景

FX 导出的 model.py 使用 `torch.ops.aten._assert_scalar.default()`，但运行时传入的是 Tensor 而非 Python bool，导致错误：

```
RuntimeError: aten::_assert_scalar() Expected a value of type 'number' for argument 'self' but instead found type 'Tensor'.
```

### 问题代码示例

```python
le_2 = item <= s0;  item = None
_assert_scalar_default = torch.ops.aten._assert_scalar.default(le_2, "Runtime assertion failed for expression u0 <= s0 on node 'le_2'")
```

`item <= s0` 返回 Tensor/SymBool，但 `_assert_scalar` 期望 Python bool。

### 修复方法

替换 `_assert_scalar` 为兼容写法：

```python
# 修复前
_assert_scalar_default = torch.ops.aten._assert_scalar.default(le_2, "Runtime assertion failed...")

# 修复后
if isinstance(le_2, torch.Tensor): le_2 = le_2.item()
assert le_2, "Runtime assertion failed..."
```

### 修复脚本

```python
import re

def fix_assert_scalar(model_path):
    """修复 model.py 里的 _assert_scalar API 兼容性问题"""

    with open(model_path) as f:
        content = f.read()

    pattern = r'(_assert_scalar_default\w*\s*=\s*)torch\.ops\.aten\._assert_scalar\.default\((\w+),\s*"([^"]+)"\)'

    def replace_assert(m):
        var = m.group(2)
        msg = m.group(3)
        return f'if isinstance({var}, torch.Tensor): {var} = {var}.item(); assert {var}, "{msg}"'

    new_content = re.sub(pattern, replace_assert, content)

    with open(model_path, "w") as f:
        f.write(new_content)

    print(f"Fixed _assert_scalar in {model_path}")

# 使用
fix_assert_scalar("/path/to/subgraph/model.py")
```

### 验证

```bash
python3.10 /ssd1/liangtai-work/DimGeneralizeAgent/scripts/ops.py verify /path/to/subgraph
# 输出: OK: runs successfully
```