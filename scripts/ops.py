#!/usr/bin/env python3
"""
操作原语: 对单个子图执行维度泛化的底层操作。

Agent 应根据 diagnose.py 的诊断结果，选择合适的操作原语组合执行。
不要盲目批量执行，而是先分析再行动。

可用操作:
    gen-constraints   - 为含 SymInt 的子图生成 input_tensor_constraints.py
    symbolize         - 对硬编码维度的子图执行 FX Pass 符号化
    assign-reifier    - 为子图匹配 Reifier 并写入 graph_net.json
    reify-preview     - 预览 Reifier 会返回的 9 组维度值
    generalize        - 为子图生成 9 份维度泛化副本
    verify            - 验证泛化后的子图是否可运行

用法:
    # 诊断
    python3.10 ops.py diagnose /ssd1/liangtai-work/lt_submit4.7/01-ai_Yi-1.5-6B-Chat/subgraph_3

    # 生成约束
    python3.10 ops.py gen-constraints /ssd1/liangtai-work/lt_submit4.7/01-ai_Yi-1.5-6B-Chat/subgraph_3

    # 预览 reifier 结果
    python3.10 ops.py reify-preview /ssd1/liangtai-work/lt_submit4.7/01-ai_Yi-1.5-6B-Chat/subgraph_3

    # 分配 reifier
    python3.10 ops.py assign-reifier /ssd1/liangtai-work/lt_submit4.7/01-ai_Yi-1.5-6B-Chat/subgraph_3

    # 生成 9 份副本
    python3.10 ops.py generalize /ssd1/liangtai-work/lt_submit4.7/01-ai_Yi-1.5-6B-Chat/subgraph_3 \
        --output-dir /ssd1/liangtai-work/lt_submit4.7_dim_gen

    # 批量处理（慎用，建议先诊断再批量）
    python3.10 ops.py batch /ssd1/liangtai-work/lt_submit4.7 \
        --action gen-constraints --status-filter needs_constraints
"""

import argparse
import json
import os
import sys
import re
import logging
import copy
import functools
import shutil
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, "/ssd1/liangtai-work/GraphNet")

from graph_net.dynamic_dim_constraints import DynamicDimConstraints
import graph_net.graph_net_json_file_util as gn_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============== Helper functions ==============

def parse_weight_meta(wm_path):
    if not os.path.exists(wm_path):
        return []
    with open(wm_path) as f:
        content = f.read()
    tensors = []
    for match in re.finditer(r'class\s+\w+:\s*\n((?:\s+\w+\s*=\s*.*\n)*)', content, re.MULTILINE):
        name = shape = dtype = data = None
        for line in match.group(1).strip().split('\n'):
            line = line.strip()
            if line.startswith('name'):
                name = line.split('=', 1)[1].strip().strip('"')
            elif line.startswith('shape'):
                shape = eval(line.split('=', 1)[1].strip())
            elif line.startswith('dtype'):
                dtype = line.split('=', 1)[1].strip().strip('"')
            elif line.startswith('data'):
                try: data = eval(line.split('=', 1)[1].strip())
                except: pass
        if name is not None:
            tensors.append({"name": name, "shape": shape if shape is not None else [], "dtype": dtype, "data": data})
    return tensors


def parse_forward_signature(sg_path):
    """解析 forward 签名，返回 (symint_params, tensor_params)"""
    model_path = os.path.join(sg_path, "model.py")
    with open(model_path) as f:
        content = f.read()
    m = re.search(r'def forward\(self,\s*(.*?)\):', content, re.DOTALL)
    if not m:
        return [], []
    sig = m.group(1)
    symint_params = re.findall(r'(s\d+)\s*:\s*torch\.SymInt', sig)
    tensor_params = re.findall(r'(L_\w+)\s*:\s*torch\.\w+', sig)
    return symint_params, tensor_params


def update_tensor_metas_by_dyn_dim_cstr(tensor_metas, dyn_dim_cstr):
    input_shapes_with_names = dyn_dim_cstr.input_shapes
    name2shape = {
        name: [dyn_dim_cstr._try_reify(dim) for dim in shape]
        for shape, name in input_shapes_with_names
    }
    for tensor_meta in tensor_metas:
        if tensor_meta.name not in name2shape:
            continue
        tensor_meta.shape = name2shape[tensor_meta.name]
        if tensor_meta.data is not None:
            assert isinstance(tensor_meta.data, (list, tuple))
            size = functools.reduce(lambda a, b: a * b, tensor_meta.shape, 1)
            extended = list(tensor_meta.data)
            while len(extended) < size:
                extended.extend(extended)
            tensor_meta.data = extended[:size]


# ============== Operations ==============

def op_diagnose(sg_path):
    """诊断单个子图"""
    from graph_net.torch.reifier_factory import ReifierFactory

    symint_params, tensor_params = parse_forward_signature(sg_path)
    tensors = parse_weight_meta(os.path.join(sg_path, "weight_meta.py"))

    print(f"Subgraph: {sg_path}")
    print(f"  SymInt params: {symint_params}")
    print(f"  Tensor params: {len(tensor_params)}")

    # Show first tensor shape
    tensor_dict = {t["name"]: t for t in tensors}
    for tp in tensor_params[:3]:
        if tp in tensor_dict:
            print(f"  {tp}: shape={tensor_dict[tp]['shape']}, dtype={tensor_dict[tp]['dtype']}")

    # SymInt example values
    for s in symint_params:
        if s in tensor_dict:
            print(f"  {s}: example_value={tensor_dict[s]['data']}")

    # Check constraint status
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    if os.path.exists(cstr_path):
        with open(cstr_path) as f:
            content = f.read()
        if "Symbol" in content and "dynamic_dim_constraint_symbols" in content:
            try:
                cstr = DynamicDimConstraints.unserialize_from_py_file(cstr_path)
                print(f"  Constraints: valid, symbols={[s.name for s in cstr.symbols]}")
                print(f"  Symbolic shapes: {cstr.serialize_symbolic_input_shapes_to_str()}")
            except Exception as e:
                print(f"  Constraints: parse error: {e}")
        elif len(content.strip()) == 0:
            print(f"  Constraints: empty")
        else:
            print(f"  Constraints: invalid")
    else:
        print(f"  Constraints: missing")

    # Check reifier
    json_path = os.path.join(sg_path, "graph_net.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            gn = json.load(f)
        reifier = gn.get("symbolic_dimension_reifier")
        print(f"  Reifier: {reifier or 'not assigned'}")
        print(f"  dynamic: {gn.get('dynamic')}")

    # Try matching reifier
    if symint_params:
        try:
            factory = ReifierFactory(config={}, model_path=sg_path)
            matched = factory.get_matched_reifier_name()
            print(f"  Would match reifier: {matched}")
        except Exception as e:
            print(f"  Reifier matching error: {e}")


def op_gen_constraints(sg_path):
    """生成 input_tensor_constraints.py（符号化）

    符号化：分析 tensor shapes，找公共维度，用 sympy Symbol 标记。
    与 SymInt 无关，适用于有 SymInt 和无 SymInt 的子图。
    """
    symint_params, tensor_params = parse_forward_signature(sg_path)
    tensors = parse_weight_meta(os.path.join(sg_path, "weight_meta.py"))
    tensor_dict = {t["name"]: t for t in tensors}

    # 收集所有输入 tensor 的 input_shapes
    input_shapes = []
    dynamic_input_names = []  # 非权重的输入 Tensor
    for t in tensors:
        if t["name"] in tensor_params and len(t["shape"]) > 0:
            input_shapes.append((list(t["shape"]), t["name"]))
            is_weight = "_parameters_" in t["name"]
            if not is_weight:
                dynamic_input_names.append(t["name"])

    if not input_shapes:
        print(f"SKIP: No input tensors with shape in {sg_path}")
        return False

    dyn_dim_cstrs = DynamicDimConstraints.make_by_named_inputs(input_shapes)

    # 确定哪些维度是动态的
    # 策略：分析动态输入 tensor 的 shapes，找出公共维度（高频维度）
    dynamic_set = set(dynamic_input_names)

    # 如果有 SymInt，用 SymInt example_value 识别动态维度
    symint_values = set()
    if symint_params:
        for t in tensors:
            if t["name"] in symint_params and t.get("data") is not None:
                for v in (t["data"] if isinstance(t["data"], list) else [t["data"]]):
                    if isinstance(v, int) and v > 1:
                        symint_values.add(v)

    # 分析动态输入的维度频率
    from collections import defaultdict, Counter
    dim_freq = Counter()  # 维度值 -> 出现次数
    dim_positions = defaultdict(list)  # 维度值 -> [(idx, axis, name)]

    for idx, (shape, name) in enumerate(dyn_dim_cstrs.input_shapes):
        if name not in dynamic_set:
            continue
        for axis, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 1:
                dim_freq[dim] += 1
                dim_positions[dim].append((idx, axis, name))

    # 确定 candidate 维度值
    # 如果有 SymInt，优先使用 SymInt example_value
    # 否则，使用高频维度（出现次数 >= 2）
    candidate_dims = set()
    if symint_values:
        candidate_dims = symint_values
    else:
        # 找出高频维度
        for dim, count in dim_freq.items():
            if count >= 2:
                candidate_dims.add(dim)
        # 如果没有高频维度，取出现次数最多的维度
        if not candidate_dims and dim_freq:
            top_dim = dim_freq.most_common(1)[0][0]
            candidate_dims.add(top_dim)

    if not candidate_dims:
        print(f"SKIP: No dynamic dimensions found in {sg_path}")
        return False

    # 符号化：按 axis 优先级处理
    # axis 1 (seq_len) 优先，然后 axis 0 (batch)，然后其他
    def axis_priority(item):
        axis = item[0][0]
        if axis == 1: return 0
        if axis == 0: return 1
        return 2

    from collections import defaultdict
    dim_groups = defaultdict(list)  # (axis, dim_value) -> [(input_idx, axis, name)]
    for dim_val in candidate_dims:
        for idx, axis, name in dim_positions.get(dim_val, []):
            dim_groups[(axis, dim_val)].append((idx, axis, name))

    for (axis, dim_val), positions in sorted(dim_groups.items(), key=axis_priority):
        target_positions = set((idx, ax) for idx, ax, name in positions)
        sym = dyn_dim_cstrs.symbolize(
            filter_fn=lambda name, idx, axis, dim, _pos=target_positions, _dim=dim_val:
                (idx, axis) in _pos and dim == _dim
        )

    if len(dyn_dim_cstrs.symbols) == 0:
        print(f"SKIP: No dimensions could be symbolized in {sg_path}")
        return False

    cstr_code = dyn_dim_cstrs.serialize_to_py_str()
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    with open(cstr_path, "w") as f:
        f.write(cstr_code)

    # 如果有 SymInt，更新 weight_meta.py 里的 SymInt example_value
    if symint_params:
        wm_path = os.path.join(sg_path, "weight_meta.py")
        if os.path.exists(wm_path):
            with open(wm_path) as f:
                wm_content = f.read()

            for sym in dyn_dim_cstrs.symbols:
                sym_name = str(sym).lower()  # S0 -> s0
                example_val = dyn_dim_cstrs.symbol2example_value.get(sym, 4)
                pattern = rf'(class\s+\w+:\s*\n(?:\s+\w+\s*=\s*.*\n)*?\s+name\s*=\s*"{sym_name}"\s*\n(?:\s+\w+\s*=\s*.*\n)*?)\s+data\s*=\s*\[.*?\]'
                replacement = rf'\1\n\tdata = [{example_val}]'
                wm_content = re.sub(pattern, replacement, wm_content)

            with open(wm_path, "w") as f:
                f.write(wm_content)

    print(f"OK: Generated constraints for {sg_path}")
    print(f"  Has SymInt: {bool(symint_params)}")
    print(f"  Symbols: {[s.name for s in dyn_dim_cstrs.symbols]}")
    print(f"  Symbolic shapes: {dyn_dim_cstrs.serialize_symbolic_input_shapes_to_str()}")
    print(f"  Example values: {dyn_dim_cstrs.symbol2example_value}")
    return True


def op_symbolize(sg_path):
    """SymInt 参数化：基于约束文件，给 model.py 添加 SymInt 参数

    仅用于 hardcoded 子图（没有 SymInt 参数的子图）。
    需要先运行 gen-constraints 生成约束文件。

    注意：此操作使用 GraphNet 的 DimensionSymbolizer，它会：
    1. 读取 input_tensor_constraints.py
    2. 分析哪些维度应该变为 SymInt
    3. 重写 model.py，将硬编码维度替换为 SymInt 参数
    """
    from graph_net.torch.sample_pass.dimension_symbolizer import DimensionSymbolizer

    symint_params, _ = parse_forward_signature(sg_path)
    if symint_params:
        print(f"SKIP: Already has SymInt in {sg_path}")
        return False

    # 检查约束文件是否存在
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    if not os.path.exists(cstr_path):
        print(f"SKIP: No constraints file in {sg_path} (run gen-constraints first)")
        return False

    # 检查约束文件是否有效
    with open(cstr_path) as f:
        content = f.read()
    if not ("Symbol" in content and "dynamic_dim_constraint_symbols" in content):
        print(f"SKIP: Invalid constraints file in {sg_path}")
        return False

    try:
        symbolizer = DimensionSymbolizer(config={
            "model_path_prefix": os.path.dirname(os.path.dirname(sg_path)),
            "output_dir": os.path.dirname(sg_path),
        })
        rel_path = os.path.basename(os.path.dirname(sg_path)) + "/" + os.path.basename(sg_path)
        symbolizer(rel_path)
        print(f"OK: Symbolized (added SymInt) for {sg_path}")
        return True
    except Exception as e:
        print(f"ERROR: Symbolization failed for {sg_path}: {e}")
        return False


def op_assign_reifier(sg_path):
    """为子图匹配 Reifier 并写入 graph_net.json"""
    from graph_net.torch.reifier_factory import ReifierFactory

    # 先检查约束文件
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    if os.path.exists(cstr_path):
        with open(cstr_path) as f:
            content = f.read()
        if not ("Symbol" in content and "dynamic_dim_constraint_symbols" in content):
            print(f"SKIP: No valid constraints in {sg_path} (run gen-constraints first)")
            return False
    else:
        print(f"SKIP: No constraints file in {sg_path} (run gen-constraints first)")
        return False

    # 检查是否已有 reifier
    json_path = os.path.join(sg_path, "graph_net.json")
    with open(json_path) as f:
        gn = json.load(f)
    if gn.get("symbolic_dimension_reifier"):
        print(f"SKIP: Already has reifier '{gn['symbolic_dimension_reifier']}' in {sg_path}")
        return False

    # 匹配
    factory = ReifierFactory(config={}, model_path=sg_path)
    matched = factory.get_matched_reifier_name()

    if matched is None:
        # 获取形状信息帮助诊断
        try:
            cstr = DynamicDimConstraints.unserialize_from_py_file(cstr_path)
            shapes_str = cstr.serialize_symbolic_input_shapes_to_str()
            print(f"NO_MATCH: No reifier for shapes {shapes_str} in {sg_path}")
        except Exception as e:
            print(f"NO_MATCH: Cannot read constraints: {e}")
        return False

    # 写入
    gn_json.update_json(sg_path, gn_json.kSymbolicDimensionReifier, matched)
    print(f"OK: Assigned reifier '{matched}' to {sg_path}")
    return True


def op_reify_preview(sg_path):
    """预览 Reifier 会返回的 9 组维度值"""
    from graph_net.torch.sym_dim_reifiers.reifier_mgr import get_reifier

    json_path = os.path.join(sg_path, "graph_net.json")
    with open(json_path) as f:
        gn = json.load(f)

    reifier_name = gn.get("symbolic_dimension_reifier")
    if not reifier_name:
        print(f"SKIP: No reifier assigned in {sg_path} (run assign-reifier first)")
        return

    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    dyn_dim_cstrs = DynamicDimConstraints.unserialize_from_py_file(cstr_path)

    reifier_class = get_reifier(reifier_name)
    reifier_instance = reifier_class(sg_path)

    if not reifier_instance.match():
        print(f"ERROR: Reifier '{reifier_name}' does not match {sg_path}")
        return

    result = reifier_instance.reify()
    print(f"Reifier: {reifier_name}")
    print(f"Current symbolic shapes: {dyn_dim_cstrs.serialize_symbolic_input_shapes_to_str()}")
    print(f"Current example values: {dyn_dim_cstrs.symbol2example_value}")
    print(f"\n9 dimension variants:")

    for key, values in result.items():
        key_str = str(key)
        if isinstance(key, tuple):
            key_str = "(" + ", ".join(str(k) for k in key) + ")"
        print(f"  Symbols: {key_str}")
        for i, v in enumerate(values):
            print(f"    [{i}] {v}")


def op_llm_reify(sg_path, values_str=None):
    """LLM 具体化：提取符号信息供 Agent 推理，或写入 Agent 推理的维度值

    用法:
        # 1. 提取符号信息（输出给 Agent 推理）
        python3.10 ops.py llm-reify <sg_path>

        # 2. 写入 Agent 推理的维度值
        python3.10 ops.py llm-reify <sg_path> --values "S0=64,128,256 S1=1,1,1"
    """
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")

    if not os.path.exists(cstr_path):
        print(f"ERROR: No constraints file in {sg_path}")
        print("Run gen-constraints first.")
        return False

    try:
        cstr = DynamicDimConstraints.unserialize_from_py_file(cstr_path)
    except Exception as e:
        print(f"ERROR: Failed to parse constraints: {e}")
        return False

    if len(cstr.symbols) == 0:
        print(f"ERROR: No symbols found in {sg_path}")
        return False

    # 提取符号信息
    symbols = [str(s) for s in cstr.symbols]
    example_values = {str(k): v for k, v in cstr.symbol2example_value.items()}
    symbolic_shapes = cstr.serialize_symbolic_input_shapes_to_str()

    # 分析输入 tensor 信息
    input_shapes = []
    for shape, name in cstr.input_shapes:
        shape_str = [str(d) for d in shape]
        input_shapes.append({"name": name, "shape": shape_str})

    if values_str is None:
        # 输出符号信息供 Agent 推理
        print("\n" + "=" * 60)
        print("符号化信息（供 Agent 推理具体维度值）")
        print("=" * 60)

        print(f"\n## 子图路径: {sg_path}")
        print(f"\n## 符号: {', '.join(symbols)}")
        print(f"\n## 当前示例值: {example_values}")
        print(f"\n## 符号化形状: {symbolic_shapes}")
        print(f"\n## 输入 Tensor:")
        for inp in input_shapes[:5]:
            print(f"  - {inp['name']}: {inp['shape']}")
        if len(input_shapes) > 5:
            print(f"  - ... 还有 {len(input_shapes) - 5} 个")

        print("\n" + "-" * 60)
        print("请 Agent 根据上述信息，推理出每个符号的推荐维度值。")
        print("返回格式: S0=64,128,256 S1=1,1,1")
        print("（每个符号提供多组值，组数应相同，如都是 9 组）")
        print("-" * 60)

        # 输出 JSON 格式
        result = {
            "path": sg_path,
            "symbols": symbols,
            "example_values": example_values,
            "symbolic_shapes": symbolic_shapes,
            "input_shapes": input_shapes,
        }
        print("\n## JSON 格式:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        return True

    else:
        # 解析并写入 Agent 推理的值
        values = {}
        for part in re.split(r'[;\s]+', values_str.strip()):
            if '=' in part:
                sym, vals = part.split('=', 1)
            elif ':' in part:
                sym, vals = part.split(':', 1)
            else:
                continue
            sym = sym.strip()
            vals = [int(v.strip()) for v in vals.split(',') if v.strip().isdigit()]
            if vals:
                values[sym] = vals

        if not values:
            print(f"ERROR: Failed to parse values from: {values_str}")
            print("Expected format: 'S0=64,128,256 S1=1,1,1'")
            return False

        # 验证符号
        for sym in values:
            if sym not in symbols:
                print(f"WARNING: Unknown symbol: {sym}")

        # 检查各组长度是否一致
        lengths = [len(v) for v in values.values()]
        if len(set(lengths)) > 1:
            print(f"ERROR: Inconsistent value counts: {dict((k, len(v)) for k, v in values.items())}")
            print("All symbols must have the same number of values")
            return False

        num_variants = lengths[0] if lengths else 0
        print(f"\n将写入 {num_variants} 组维度值:")
        for sym, vals in values.items():
            print(f"  {sym}: {vals}")

        # 写入 llm_reified_values.json
        output = {
            "symbols": list(values.keys()),
            "values": values,
            "num_variants": num_variants,
        }

        output_path = os.path.join(sg_path, "llm_reified_values.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nOK: Written to {output_path}")
        print(f"Next: run generalize with --use-llm flag")
        return True


def op_generalize(sg_path, output_dir, data_dir, dim_indices=None, resume=True, use_llm=False):
    """为子图生成维度泛化副本

    支持两种方式：
    1. Reifier 方式：使用预设的 Reifier（默认）
    2. LLM 方式：使用 Agent 推理的维度值（--use-llm）
    """
    from graph_net.torch.sym_dim_reifiers.reifier_mgr import get_reifier
    from graph_net.tensor_meta import TensorMeta
    from graph_net.hash_util import get_sha256_hash

    # 获取约束
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    dyn_dim_cstrs = DynamicDimConstraints.unserialize_from_py_file(cstr_path)

    symbols = [str(s) for s in dyn_dim_cstrs.symbols]
    reified_dims = []

    if use_llm:
        # 使用 LLM 具体化的维度值
        llm_path = os.path.join(sg_path, "llm_reified_values.json")
        if not os.path.exists(llm_path):
            print(f"ERROR: No LLM values file in {sg_path}")
            print("Run llm-reify with --values first.")
            return False

        with open(llm_path) as f:
            llm_data = json.load(f)

        llm_symbols = llm_data["symbols"]
        llm_values = llm_data["values"]
        num_variants = llm_data["num_variants"]

        # 生成 reified_dims 格式
        for i in range(num_variants):
            dims = [llm_values[s][i] for s in llm_symbols]
            reified_dims.append(dims)

        print(f"Using LLM-reified values: {num_variants} variants")
        if dim_indices is None:
            dim_indices = list(range(num_variants))

    else:
        # 使用 Reifier 方式
        if dim_indices is None:
            dim_indices = list(range(9))

        json_path = os.path.join(sg_path, "graph_net.json")
        with open(json_path) as f:
            gn = json.load(f)
        reifier_name = gn.get("symbolic_dimension_reifier")
        if not reifier_name:
            print(f"SKIP: No reifier in {sg_path}")
            return False

        reifier_class = get_reifier(reifier_name)
        reifier_instance = reifier_class(sg_path)
        assert reifier_instance.match()
        symbols2reified_dims = reifier_instance.reify()
        assert len(symbols2reified_dims) == 1
        key, reified_dims = next(iter(symbols2reified_dims.items()))

        # Normalize key
        if isinstance(key, tuple):
            symbols = list(key)
        else:
            symbols = [key]
            reified_dims = [[v] for v in reified_dims]

    # 获取 rel_path
    rel_path = os.path.relpath(sg_path, data_dir)

    # Load tensor metas
    tensor_metas = []
    for meta_file in ["input_meta.py", "weight_meta.py"]:
        meta_path = os.path.join(sg_path, meta_file)
        if os.path.exists(meta_path):
            tensor_metas.extend(TensorMeta.unserialize_from_py_file(meta_path))

    generated = []
    for idx, dims in enumerate(reified_dims):
        if idx not in dim_indices:
            continue

        out_path = os.path.join(output_dir, str(idx), rel_path)
        if resume and os.path.exists(os.path.join(out_path, "model.py")):
            generated.append(idx)
            continue

        os.makedirs(out_path, exist_ok=True)
        shutil.copytree(Path(sg_path), Path(out_path), dirs_exist_ok=True)

        # Update constraints
        symbol2example_value = OrderedDict(list(zip(symbols, dims)))
        cur_dyn_dim_cstrs = copy.deepcopy(dyn_dim_cstrs)
        cur_tensor_metas = copy.deepcopy(tensor_metas)
        cur_dyn_dim_cstrs.update_symbol2example_value(symbol2example_value)
        update_tensor_metas_by_dyn_dim_cstr(cur_tensor_metas, cur_dyn_dim_cstrs)

        # Write updated files
        Path(os.path.join(out_path, "input_tensor_constraints.py")).write_text(
            cur_dyn_dim_cstrs.serialize_to_py_str()
        )

        # Write updated tensor metas back to weight_meta.py and input_meta.py
        # (key fix: previously only SymInt data was updated via regex,
        #  but Tensor shapes were updated in memory and never written back)
        input_meta_names = set()
        for meta_file in ["input_meta.py"]:
            meta_path = os.path.join(sg_path, meta_file)
            if os.path.exists(meta_path):
                for tm in TensorMeta.unserialize_from_py_file(meta_path):
                    input_meta_names.add(tm.name)

        input_metas = [tm for tm in cur_tensor_metas if tm.name in input_meta_names]
        weight_metas = [tm for tm in cur_tensor_metas if tm.name not in input_meta_names]

        if input_metas:
            TensorMeta.save_tensor_metas(os.path.join(out_path, "input_meta.py"), input_metas)
        if weight_metas:
            TensorMeta.save_tensor_metas(os.path.join(out_path, "weight_meta.py"), weight_metas)

        # Update SymInt data values in weight_meta.py
        wm_path = os.path.join(out_path, "weight_meta.py")
        if os.path.exists(wm_path):
            with open(wm_path) as f:
                wm_content = f.read()
            for sym, value in symbol2example_value.items():
                sym_name = str(sym).lower()  # S0 -> s0
                pattern = rf'(class\s+\w+:\s*\n(?:\s+\w+\s*=\s*.*\n)*?\s+name\s*=\s*"{sym_name}"\s*\n(?:\s+\w+\s*=\s*.*\n)*?)\s+data\s*=\s*\[.*?\]'
                replacement = rf'\1\n\tdata = [{value}]'
                wm_content = re.sub(pattern, replacement, wm_content)
            with open(wm_path, "w") as f:
                f.write(wm_content)

        # Fix missing imports in model.py (e.g. math_floor used but not imported)
        model_py_path = os.path.join(out_path, "model.py")
        if os.path.exists(model_py_path):
            model_code = Path(model_py_path).read_text()
            patches = []
            if "math_floor" in model_code and "from math import floor as math_floor" not in model_code and "import math" not in model_code:
                patches.append("from math import floor as math_floor")
            if patches:
                # Insert after existing imports
                import_line = "\n".join(patches) + "\n"
                model_code = model_code.replace("import torch\n", "import torch\n" + import_line, 1)
                Path(model_py_path).write_text(model_code)

        # Update graph_hash
        model_code = Path(os.path.join(out_path, "model.py")).read_text()
        Path(os.path.join(out_path, "graph_hash.txt")).write_text(get_sha256_hash(model_code))

        generated.append(idx)

    print(f"OK: Generated dims {generated} for {sg_path}")
    return True


def op_verify(sg_path):
    """验证子图是否可运行（使用 GraphNet 的 run_model 执行 forward）"""
    import subprocess
    try:
        result = subprocess.run(
            ["python3.10", "-m", "graph_net.torch.run_model", "--model-path", sg_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/ssd1/liangtai-work/GraphNet"
        )
        if result.returncode == 0:
            print(f"OK: {sg_path} runs successfully")
            return True
        else:
            print(f"ERROR: {sg_path} failed:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"ERROR: {sg_path} timed out (60s)")
        return False
    except Exception as e:
        print(f"ERROR: {sg_path} failed: {e}")
        return False


# ============== Batch mode ==============

def op_batch(data_dir, action, status_filter=None, model_name=None, output_dir=None, resume=True, dim_indices=None, diag_json=None):
    """批量执行某个操作"""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from diagnose import find_all_subgraphs, diagnose_subgraph

    subgraphs = find_all_subgraphs(data_dir)
    if model_name:
        subgraphs = [s for s in subgraphs if model_name in s]

    if status_filter:
        # 优先使用缓存的诊断结果
        if diag_json and os.path.exists(diag_json):
            logger.info(f"Loading diagnosis from {diag_json}...")
            with open(diag_json) as f:
                cached = json.load(f)
            # 支持两种格式：列表 [...] 或字典 {"subgraphs": [...]}
            if isinstance(cached, list):
                items = cached
            else:
                items = cached.get("subgraphs", [])
            cached_map = {item["path"]: item["status"] for item in items}
            filtered = [s for s in subgraphs if cached_map.get(s) == status_filter]
            logger.info(f"After filter (cached): {len(filtered)} subgraphs")
            subgraphs = filtered
        else:
            logger.info(f"Filtering by status={status_filter} (re-diagnosing each)...")
            filtered = []
            for sg in subgraphs:
                diag = diagnose_subgraph(sg)
                if diag["status"] == status_filter:
                    filtered.append(sg)
            subgraphs = filtered
            logger.info(f"After filter: {len(subgraphs)} subgraphs")

    ok = skip = err = 0
    for i, sg in enumerate(subgraphs):
        logger.info(f"[{i+1}/{len(subgraphs)}] {sg}")
        try:
            if action == "gen-constraints":
                result = op_gen_constraints(sg)
            elif action == "symbolize":
                result = op_symbolize(sg)
            elif action == "assign-reifier":
                result = op_assign_reifier(sg)
            elif action == "generalize":
                result = op_generalize(sg, output_dir, data_dir, dim_indices, resume)
            elif action == "verify":
                result = op_verify(sg)
            else:
                logger.error(f"Unknown action: {action}")
                return

            if result:
                ok += 1
            else:
                skip += 1
        except Exception as e:
            err += 1
            logger.error(f"  Error: {e}")

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{len(subgraphs)} (ok={ok}, skip={skip}, err={err})")

    print(f"\nBatch {action} complete: ok={ok}, skip={skip}, err={err}")


# ============== Snapshot ==============

def op_snapshot():
    """扫描三个数据目录，更新 PROGRESS.md"""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from diagnose import find_all_subgraphs, diagnose_subgraph

    dirs_info = [
        ("lt_submit4.7", "/ssd1/liangtai-work/lt_submit4.7", "/ssd1/liangtai-work/lt_submit4.7_dim_gen"),
        ("lt_submit4.9", "/ssd1/liangtai-work/lt_submit4.9", "/ssd1/liangtai-work/lt_submit4.9_dim_gen"),
        ("lt_submit4.3", "/ssd1/liangtai-work/lt_submit4.3", "/ssd1/liangtai-work/lt_submit4.3_dim_gen"),
        ("lt_submit4.10", "/ssd1/liangtai-work/lt_submit4.10", "/ssd1/liangtai-work/lt_submit4.10_dim_gen"),
        ("lt_submit4.13", "/ssd1/liangtai-work/lt_submit4.13", "/ssd1/liangtai-work/lt_submit4.13_dim_gen"),
    ]

    results = []
    for name, data_dir, output_dir in dirs_info:
        logger.info(f"Scanning {name}...")
        subgraphs = find_all_subgraphs(data_dir)
        total = len(subgraphs)
        counts = {"hardcoded": 0, "needs_symbolize": 0, "needs_constraints": 0, "needs_reifier": 0,
                  "ready_for_generalization": 0, "broken": 0}
        for sg in subgraphs:
            diag = diagnose_subgraph(sg)
            status = diag["status"]
            if status in counts:
                counts[status] += 1

        # Count generalized (check output dir)
        gen_done = 0
        if os.path.exists(output_dir):
            for d in range(9):
                variant_dir = os.path.join(output_dir, str(d))
                if os.path.isdir(variant_dir):
                    gen_done += sum(1 for _, dirs, _ in os.walk(variant_dir)
                                   for sd in dirs if os.path.exists(os.path.join(variant_dir, sd, "model.py")))

        results.append({
            "name": name,
            "total": total,
            "hardcoded": counts["hardcoded"],
            "needs_symbolize": counts["needs_symbolize"],
            "needs_constraints": counts["needs_constraints"],
            "needs_reifier": counts["needs_reifier"],
            "ready": counts["ready_for_generalization"],
            "broken": counts["broken"],
            "gen_done": gen_done,
        })

    # Determine current task
    # 处理顺序: gen-constraints → symbolize → assign-reifier → generalize
    all_done = True
    current_task = ""
    for r in results:
        if r["needs_constraints"] > 0:
            current_task = f"{r['name']}: gen-constraints (还有 {r['needs_constraints']} 个子图需要生成约束)"
            all_done = False
            break
        if r["hardcoded"] > 0:
            # hardcoded 需要先执行 gen-constraints
            current_task = f"{r['name']}: gen-constraints (还有 {r['hardcoded']} 个硬编码子图需要生成约束)"
            all_done = False
            break
        if r["needs_symbolize"] > 0:
            current_task = f"{r['name']}: symbolize (还有 {r['needs_symbolize']} 个子图需要 SymInt 参数化)"
            all_done = False
            break
        if r["needs_reifier"] > 0:
            current_task = f"{r['name']}: assign-reifier (还有 {r['needs_reifier']} 个子图需要分配 Reifier)"
            all_done = False
            break
        if r["ready"] > 0:
            current_task = f"{r['name']}: generalize (还有 {r['ready']} 个子图可以泛化)"
            all_done = False
            break
    if all_done:
        current_task = "全部完成"

    # Write PROGRESS.md
    progress_path = os.path.join(os.path.dirname(scripts_dir), "PROGRESS.md")

    main_rows = []
    total_all = total_hc = total_ns = total_nc = total_nr = total_gen = 0
    for r in results:
        status = "完成" if (r["hardcoded"] == 0 and r["needs_symbolize"] == 0 and r["needs_constraints"] == 0 and r["needs_reifier"] == 0
                          and r["ready"] == 0) else "进行中" if r["gen_done"] > 0 else "未开始"
        main_rows.append(f"| {r['name']} | {r['total']} | {r['hardcoded']} | {r['needs_symbolize']} | {r['needs_constraints']} | {r['needs_reifier']} | {r['gen_done']} | {status} |")
        total_all += r["total"]
        total_hc += r["hardcoded"]
        total_ns += r["needs_symbolize"]
        total_nc += r["needs_constraints"]
        total_nr += r["needs_reifier"]
        total_gen += r["gen_done"]
    main_rows.append(f"| **合计** | **{total_all}** | **{total_hc}** | **{total_ns}** | **{total_nc}** | **{total_nr}** | **{total_gen}** | |")

    broken_rows = [f"| {r['name']} | {r['broken']} |" for r in results]

    content = f"""# 维度泛化进度

> Agent 启动后先读本文件。只看"当前任务"和"各目录进度"两节即可开始工作。

## 当前任务

**{current_task}**

## 各目录进度

处理顺序: 4.10 → 4.7 → 4.9 → 4.13 → 4.3（从小到大）

| 目录 | 子图数 | 硬编码 | 需symbolize | 需gen-constraints | 需assign-reifier | 已generalize | 状态 |
|------|--------|--------|-------------|-------------------|------------------|-------------|------|
{chr(10).join(main_rows)}

## 损坏子图（跳过）

| 目录 | 数量 |
|------|------|
{chr(10).join(broken_rows)}

## 处理规则

1. 处理流程: gen-constraints (所有子图) → symbolize (仅硬编码子图) → assign-reifier → generalize
2. 每步完成后运行 `python3.10 scripts/ops.py snapshot` 更新本文件
3. 遇到大量错误停下来分析，不要继续刷

## 路径速查

| | 数据 | 输出 |
|--|------|------|
| 4.10 | `/ssd1/liangtai-work/lt_submit4.10` | `/ssd1/liangtai-work/lt_submit4.10_dim_gen` |
| 4.7 | `/ssd1/liangtai-work/lt_submit4.7` | `/ssd1/liangtai-work/lt_submit4.7_dim_gen` |
| 4.9 | `/ssd1/liangtai-work/lt_submit4.9` | `/ssd1/liangtai-work/lt_submit4.9_dim_gen` |
| 4.13 | `/ssd1/liangtai-work/lt_submit4.13` | `/ssd1/liangtai-work/lt_submit4.13_dim_gen` |
| 4.3 | `/ssd1/liangtai-work/lt_submit4.3` | `/ssd1/liangtai-work/lt_submit4.3_dim_gen` |

## 备注

- 必须用 `python3.10`
- 环境设置: `source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh`
- 详细工具文档见 `GUIDE.md`
"""

    with open(progress_path, "w") as f:
        f.write(content)

    print(f"Progress updated: {progress_path}")
    print(f"Current task: {current_task}")
    for r in results:
        print(f"  {r['name']}: total={r['total']}, needs_cstr={r['needs_constraints']}, "
              f"needs_reifier={r['needs_reifier']}, ready={r['ready']}, "
              f"hardcoded={r['hardcoded']}, broken={r['broken']}, gen_done={r['gen_done']}")


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Dimension generalization operations")
    subparsers = parser.add_subparsers(dest="command")

    # diagnose
    p = subparsers.add_parser("diagnose", help="Diagnose a single subgraph")
    p.add_argument("sg_path")

    # gen-constraints
    p = subparsers.add_parser("gen-constraints", help="Generate input_tensor_constraints.py")
    p.add_argument("sg_path")

    # symbolize
    p = subparsers.add_parser("symbolize", help="Run FX Pass symbolization on hardcoded subgraph")
    p.add_argument("sg_path")

    # assign-reifier
    p = subparsers.add_parser("assign-reifier", help="Match and assign Reifier")
    p.add_argument("sg_path")

    # reify-preview
    p = subparsers.add_parser("reify-preview", help="Preview reifier's 9 dimension variants")
    p.add_argument("sg_path")

    # llm-reify
    p = subparsers.add_parser("llm-reify", help="LLM reification: extract symbols or apply Agent-reasoned values")
    p.add_argument("sg_path")
    p.add_argument("--values", "-v", type=str, default=None,
                   help='Agent-reasoned values, format: "S0=64,128,256 S1=1,1,1"')

    # generalize
    p = subparsers.add_parser("generalize", help="Generate dimension variants")
    p.add_argument("sg_path")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dir", required=True, help="Source data dir (for relative path calculation)")
    p.add_argument("--dim-indices", type=str, default=None, help="e.g. '0,1,2'")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--use-llm", action="store_true", help="Use LLM-reified values instead of Reifier")

    # verify
    p = subparsers.add_parser("verify", help="Verify subgraph is runnable")
    p.add_argument("sg_path")

    # batch
    p = subparsers.add_parser("batch", help="Batch execute an operation")
    p.add_argument("data_dir")
    p.add_argument("--action", required=True, choices=["gen-constraints", "symbolize", "assign-reifier", "generalize", "verify"])
    p.add_argument("--status-filter", choices=["hardcoded", "needs_symbolize", "needs_constraints", "needs_reifier", "ready_for_generalization"])
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--dim-indices", type=str, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--diag-json", type=str, default=None, help="Path to cached diagnosis JSON (from diagnose.py --output)")

    # snapshot
    p = subparsers.add_parser("snapshot", help="Scan all dirs and update PROGRESS.md")

    args = parser.parse_args()

    if args.command == "diagnose":
        op_diagnose(args.sg_path)
    elif args.command == "gen-constraints":
        op_gen_constraints(args.sg_path)
    elif args.command == "symbolize":
        op_symbolize(args.sg_path)
    elif args.command == "assign-reifier":
        op_assign_reifier(args.sg_path)
    elif args.command == "reify-preview":
        op_reify_preview(args.sg_path)
    elif args.command == "llm-reify":
        op_llm_reify(args.sg_path, args.values)
    elif args.command == "generalize":
        dim_indices = [int(x) for x in args.dim_indices.split(",")] if args.dim_indices else None
        op_generalize(args.sg_path, args.output_dir, args.data_dir, dim_indices,
                      resume=not args.no_resume, use_llm=args.use_llm)
    elif args.command == "verify":
        op_verify(args.sg_path)
    elif args.command == "batch":
        dim_indices = [int(x) for x in args.dim_indices.split(",")] if args.dim_indices else list(range(9))
        op_batch(args.data_dir, args.action, args.status_filter, args.model_name,
                 args.output_dir, resume=not args.no_resume, dim_indices=dim_indices,
                 diag_json=args.diag_json)
    elif args.command == "snapshot":
        op_snapshot()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
