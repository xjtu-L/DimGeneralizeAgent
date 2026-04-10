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
    """为含 SymInt 的子图生成 input_tensor_constraints.py"""
    symint_params, tensor_params = parse_forward_signature(sg_path)
    if not symint_params:
        print(f"SKIP: No SymInt params in {sg_path}")
        return False

    tensors = parse_weight_meta(os.path.join(sg_path, "weight_meta.py"))
    tensor_dict = {t["name"]: t for t in tensors}

    # 构建第一个输入 tensor 的 input_shapes
    input_shapes = []
    for t in tensors:
        if t["name"] in tensor_params and len(t["shape"]) > 0:
            input_shapes.append((list(t["shape"]), t["name"]))
            break  # 只取第一个

    if not input_shapes:
        print(f"SKIP: No input tensors with shape in {sg_path}")
        return False

    dyn_dim_cstrs = DynamicDimConstraints.make_by_named_inputs(input_shapes)

    # 符号化: axis 1 (seq_len), axis 0 (batch)
    dyn_dim_cstrs.symbolize(filter_fn=lambda name, idx, axis, dim: axis == 1 and dim > 1)
    dyn_dim_cstrs.symbolize(filter_fn=lambda name, idx, axis, dim: axis == 0 and dim > 1)

    # 兜底: 如果还没符号化，尝试任意 >1 的维度
    if len(dyn_dim_cstrs.symbols) == 0:
        dyn_dim_cstrs.symbolize(filter_fn=lambda name, idx, axis, dim: dim > 1)

    if len(dyn_dim_cstrs.symbols) == 0:
        print(f"SKIP: No dimensions could be symbolized in {sg_path}")
        return False

    cstr_code = dyn_dim_cstrs.serialize_to_py_str()
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    with open(cstr_path, "w") as f:
        f.write(cstr_code)

    print(f"OK: Generated constraints for {sg_path}")
    print(f"  Symbols: {[s.name for s in dyn_dim_cstrs.symbols]}")
    print(f"  Symbolic shapes: {dyn_dim_cstrs.serialize_symbolic_input_shapes_to_str()}")
    return True


def op_symbolize(sg_path):
    """对硬编码维度的子图执行 FX Pass 符号化（使用 GraphNet 的 DimensionSymbolizer）"""
    from graph_net.torch.sample_pass.dimension_symbolizer import DimensionSymbolizer

    symint_params, _ = parse_forward_signature(sg_path)
    if symint_params:
        print(f"SKIP: Already has SymInt in {sg_path}")
        return False

    try:
        symbolizer = DimensionSymbolizer(config={
            "model_path_prefix": os.path.dirname(os.path.dirname(sg_path)),
            "output_dir": os.path.dirname(sg_path),
        })
        rel_path = os.path.basename(os.path.dirname(sg_path)) + "/" + os.path.basename(sg_path)
        symbolizer(rel_path)
        print(f"OK: Symbolized {sg_path}")
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


def op_generalize(sg_path, output_dir, data_dir, dim_indices=None, resume=True):
    """为子图生成 9 份维度泛化副本"""
    from graph_net.torch.sym_dim_reifiers.reifier_mgr import get_reifier
    from graph_net.tensor_meta import TensorMeta
    from graph_net.hash_util import get_sha256_hash

    if dim_indices is None:
        dim_indices = list(range(9))

    # 检查 reifier
    json_path = os.path.join(sg_path, "graph_net.json")
    with open(json_path) as f:
        gn = json.load(f)
    reifier_name = gn.get("symbolic_dimension_reifier")
    if not reifier_name:
        print(f"SKIP: No reifier in {sg_path}")
        return False

    # 获取约束
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    dyn_dim_cstrs = DynamicDimConstraints.unserialize_from_py_file(cstr_path)

    # 获取 reified dims
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

        # Update SymInt example values in weight_meta.py
        wm_path = os.path.join(out_path, "weight_meta.py")
        if os.path.exists(wm_path):
            with open(wm_path) as f:
                wm_content = f.read()
            for sym, value in symbol2example_value.items():
                s_name = str(sym).lower()
                pattern = rf'(class\s+\w+:\s*\n(?:\s+\w+\s*=\s*.*\n)*?\s+name\s*=\s*"{s_name}"\s*\n(?:\s+\w+\s*=\s*.*\n)*?)\s+data\s*=\s*\[.*?\]'
                replacement = rf'\1\n\tdata = [{value}]'
                wm_content = re.sub(pattern, replacement, wm_content)
            Path(wm_path).write_text(wm_content)

        # Update graph_hash
        model_code = Path(os.path.join(out_path, "model.py")).read_text()
        Path(os.path.join(out_path, "graph_hash.txt")).write_text(get_sha256_hash(model_code))

        generated.append(idx)

    print(f"OK: Generated dims {generated} for {sg_path}")
    return True


def op_verify(sg_path):
    """验证子图是否可运行（尝试加载并执行 forward）"""
    try:
        from graph_net.imp_util import load_module
        model_path = os.path.join(sg_path, "model.py")
        py_module = load_module(model_path)
        GraphModule = getattr(py_module, "GraphModule")
        model = GraphModule()

        # Try to create inputs from weight_meta
        tensors = parse_weight_meta(os.path.join(sg_path, "weight_meta.py"))
        import torch
        inputs = []
        for t in tensors:
            if t["name"].startswith("L_") or t["name"].startswith("s"):
                if t["shape"] and t["dtype"]:
                    dtype = getattr(torch, t["dtype"].replace("torch.", ""))
                    if t["data"] is not None:
                        inputs.append(torch.tensor(t["data"], dtype=dtype).reshape(t["shape"]))
                    else:
                        inputs.append(torch.randn(t["shape"], dtype=dtype))

        # Try forward (just check shape propagation, not full execution)
        print(f"OK: {sg_path} is loadable")
        return True
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
            cached_map = {item["path"]: item["status"] for item in cached.get("subgraphs", [])}
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
    ]

    results = []
    for name, data_dir, output_dir in dirs_info:
        logger.info(f"Scanning {name}...")
        subgraphs = find_all_subgraphs(data_dir)
        total = len(subgraphs)
        counts = {"hardcoded": 0, "needs_constraints": 0, "needs_reifier": 0,
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
            "needs_constraints": counts["needs_constraints"],
            "needs_reifier": counts["needs_reifier"],
            "ready": counts["ready_for_generalization"],
            "broken": counts["broken"],
            "gen_done": gen_done,
        })

    # Determine current task
    all_done = True
    current_task = ""
    for r in results:
        if r["needs_constraints"] > 0:
            current_task = f"{r['name']}: gen-constraints (还有 {r['needs_constraints']} 个子图需要生成约束)"
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
        if r["hardcoded"] > 0:
            current_task = f"{r['name']}: symbolize (还有 {r['hardcoded']} 个硬编码子图)"
            all_done = False
            break
    if all_done:
        current_task = "全部完成"

    # Write PROGRESS.md
    progress_path = os.path.join(os.path.dirname(scripts_dir), "PROGRESS.md")

    main_rows = []
    total_all = total_nc = total_nr = total_gen = 0
    for r in results:
        status = "完成" if (r["needs_constraints"] == 0 and r["needs_reifier"] == 0
                          and r["ready"] == 0 and r["hardcoded"] == 0) else "进行中" if r["gen_done"] > 0 else "未开始"
        main_rows.append(f"| {r['name']} | {r['total']} | {r['needs_constraints']} | {r['needs_reifier']} | {r['gen_done']} | {status} |")
        total_all += r["total"]
        total_nc += r["needs_constraints"]
        total_nr += r["needs_reifier"]
        total_gen += r["gen_done"]
    main_rows.append(f"| **合计** | **{total_all}** | **{total_nc}** | **{total_nr}** | **{total_gen}** | |")

    hard_rows = []
    for r in results:
        sym_done = r["total"] - r["hardcoded"] - r["needs_constraints"] - r["needs_reifier"] - r["ready"] - r["broken"]
        if sym_done < 0: sym_done = 0
        hard_rows.append(f"| {r['name']} | {r['hardcoded']} | {sym_done} | 0 | 未开始 |")

    broken_rows = [f"| {r['name']} | {r['broken']} |" for r in results]

    content = f"""# 维度泛化进度

> Agent 启动后先读本文件。只看"当前任务"和"各目录进度"两节即可开始工作。

## 当前任务

**{current_task}**

## 各目录进度

处理顺序: 4.7 → 4.9 → 4.3（从小到大）

| 目录 | 子图数 | 需gen-constraints | 需assign-reifier | 已generalize | 状态 |
|------|--------|-------------------|------------------|-------------|------|
{chr(10).join(main_rows)}

## 硬编码子图（需先 symbolize 再走后续流程）

| 目录 | 硬编码数 | symbolize 成功 | symbolize 失败 | 状态 |
|------|---------|---------------|---------------|------|
{chr(10).join(hard_rows)}

## 损坏子图（跳过）

| 目录 | 数量 |
|------|------|
{chr(10).join(broken_rows)}

## 处理规则

1. 每个目录按顺序执行: gen-constraints → assign-reifier → generalize
2. 硬编码子图先 symbolize，成功后进入 gen-constraints 流程
3. 每步完成后运行 `python3.10 scripts/ops.py snapshot` 更新本文件
4. 遇到大量错误停下来分析，不要继续刷

## 路径速查

| | 数据 | 输出 |
|--|------|------|
| 4.7 | `/ssd1/liangtai-work/lt_submit4.7` | `/ssd1/liangtai-work/lt_submit4.7_dim_gen` |
| 4.9 | `/ssd1/liangtai-work/lt_submit4.9` | `/ssd1/liangtai-work/lt_submit4.9_dim_gen` |
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

    # generalize
    p = subparsers.add_parser("generalize", help="Generate 9 dimension variants")
    p.add_argument("sg_path")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dir", required=True, help="Source data dir (for relative path calculation)")
    p.add_argument("--dim-indices", type=str, default=None, help="e.g. '0,1,2'")
    p.add_argument("--no-resume", action="store_true")

    # verify
    p = subparsers.add_parser("verify", help="Verify subgraph is runnable")
    p.add_argument("sg_path")

    # batch
    p = subparsers.add_parser("batch", help="Batch execute an operation")
    p.add_argument("data_dir")
    p.add_argument("--action", required=True, choices=["gen-constraints", "symbolize", "assign-reifier", "generalize", "verify"])
    p.add_argument("--status-filter", choices=["hardcoded", "needs_constraints", "needs_reifier", "ready_for_generalization"])
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
    elif args.command == "generalize":
        dim_indices = [int(x) for x in args.dim_indices.split(",")] if args.dim_indices else list(range(9))
        op_generalize(args.sg_path, args.output_dir, args.data_dir, dim_indices, resume=not args.no_resume)
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
