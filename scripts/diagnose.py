#!/usr/bin/env python3
"""
数据诊断工具: 扫描计算图目录，输出每个子图的结构、符号化状态、可泛化性等诊断信息。

Agent 应先运行此工具了解数据全貌，再制定针对性处理策略。

用法:
    python3.10 diagnose.py /ssd1/liangtai-work/lt_submit4.7
    python3.10 diagnose.py /ssd1/liangtai-work/lt_submit4.3 --model-name "bert-base"
    python3.10 diagnose.py /ssd1/liangtai-work/lt_submit4.9 --output /tmp/diag.json
    python3.10 diagnose.py /ssd1/liangtai-work/lt_submit4.3 --status-filter hardcoded
"""

import argparse
import json
import os
import sys
import re
import logging
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/ssd1/liangtai-work/GraphNet")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_all_subgraphs(data_dir):
    subgraphs = []
    data_path = Path(data_dir)
    for model_dir in sorted(data_path.iterdir()):
        if not model_dir.is_dir():
            continue
        has_subgraph = False
        for sub_dir in sorted(model_dir.iterdir()):
            if sub_dir.is_dir() and (sub_dir / "graph_net.json").exists():
                subgraphs.append(str(sub_dir))
                has_subgraph = True
        if not has_subgraph and (model_dir / "graph_net.json").exists():
            subgraphs.append(str(model_dir))
    return subgraphs


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


def diagnose_subgraph(sg_path):
    """对单个子图做完整诊断"""
    r = {
        "path": sg_path,
        "status": "unknown",
        "issues": [],
        "symint_params": [],
        "symint_usage": {},
        "symint_example_values": {},
        "first_tensor_name": None,
        "first_tensor_shape": None,
        "tensor_params_count": 0,
        "constraint_status": "missing",
        "constraint_symbols": [],
        "constraint_symbolic_shapes": None,
        "reifier": None,
        "dim_gen_passes": [],
        "model_name": "",
        "is_dynamic": None,
        "has_extract_py": False,
    }

    model_path = os.path.join(sg_path, "model.py")
    wm_path = os.path.join(sg_path, "weight_meta.py")
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")
    json_path = os.path.join(sg_path, "graph_net.json")
    extract_path = os.path.join(sg_path, "extract.py")

    # Check extract.py (exists in 4.3 data, means root-level model with subgraphs)
    r["has_extract_py"] = os.path.exists(extract_path)

    # 1. model.py
    if not os.path.exists(model_path):
        r["status"] = "broken"
        r["issues"].append("no model.py")
        return r

    with open(model_path) as f:
        model_content = f.read()

    m = re.search(r'def forward\(self,\s*(.*?)\):', model_content, re.DOTALL)
    if not m:
        r["status"] = "broken"
        r["issues"].append("no forward method in model.py")
        return r

    sig = m.group(1)
    symint_params = re.findall(r'(s\d+)\s*:\s*torch\.SymInt', sig)
    tensor_params = re.findall(r'(L_\w+)\s*:\s*torch\.\w+', sig)
    r["symint_params"] = symint_params
    r["tensor_params_count"] = len(tensor_params)

    # SymInt usage analysis
    for s in symint_params:
        usages = []
        for pattern, label in [
            (rf'arange\({s}[,\)]', "arange"),
            (rf'view\(\s*{s}\s*,', "view_as_batch"),
            (rf'view\([^)]*,\s*{s}\s*[,)]', "view_as_inner_dim"),
            (rf'expand\([^)]*{s}[^)]*\)', "expand"),
            (rf'slice\(None,\s*{s}', "slice_end"),
            (rf'\[:,\s*:{s}', "slice_colon"),
        ]:
            if re.search(pattern, model_content):
                usages.append(label)
        r["symint_usage"][s] = usages

    # 2. weight_meta.py
    if os.path.exists(wm_path):
        tensors = parse_weight_meta(wm_path)
        for t in tensors:
            if t["name"] in tensor_params and len(t["shape"]) > 0 and r["first_tensor_name"] is None:
                r["first_tensor_name"] = t["name"]
                r["first_tensor_shape"] = t["shape"]
            if t["name"] in symint_params and t["data"]:
                r["symint_example_values"][t["name"]] = t["data"][0] if t["data"] else None

    # 3. input_tensor_constraints.py
    if os.path.exists(cstr_path):
        try:
            with open(cstr_path) as f:
                content = f.read()
            if len(content.strip()) == 0:
                r["constraint_status"] = "empty"
            elif "Symbol" in content and "dynamic_dim_constraint_symbols" in content:
                r["constraint_status"] = "valid"
                try:
                    from graph_net.dynamic_dim_constraints import DynamicDimConstraints
                    cstr = DynamicDimConstraints.unserialize_from_py_file(cstr_path)
                    r["constraint_symbols"] = [s.name for s in cstr.symbols]
                    r["constraint_symbolic_shapes"] = cstr.serialize_symbolic_input_shapes_to_str()
                except Exception as e:
                    r["constraint_status"] = f"parse_error: {e}"
            else:
                r["constraint_status"] = "invalid"
        except Exception as e:
            r["constraint_status"] = f"read_error: {e}"
    else:
        r["constraint_status"] = "missing"

    # 4. graph_net.json
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                gn = json.load(f)
            r["reifier"] = gn.get("symbolic_dimension_reifier")
            r["dim_gen_passes"] = gn.get("dimension_generalization_passes", [])
            r["model_name"] = gn.get("model_name", "")
            r["is_dynamic"] = gn.get("dynamic")
        except:
            pass

    # 5. Determine status
    if not symint_params:
        r["status"] = "hardcoded"
    elif r["constraint_status"] == "valid":
        if r["reifier"]:
            r["status"] = "ready_for_generalization"
        else:
            r["status"] = "needs_reifier"
    else:
        r["status"] = "needs_constraints"

    return r


def main():
    parser = argparse.ArgumentParser(description="Diagnose subgraphs for dimension generalization")
    parser.add_argument("data_dir", help="Data directory to scan")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save full results to JSON")
    parser.add_argument("--status-filter", type=str, default=None,
                        choices=["hardcoded", "needs_constraints", "needs_reifier", "ready_for_generalization", "broken"])
    args = parser.parse_args()

    subgraphs = find_all_subgraphs(args.data_dir)
    if args.model_name:
        subgraphs = [s for s in subgraphs if args.model_name in s]

    logger.info(f"Diagnosing {len(subgraphs)} subgraphs in {args.data_dir}...")

    results = []
    for i, sg in enumerate(subgraphs):
        results.append(diagnose_subgraph(sg))
        if (i + 1) % 200 == 0:
            logger.info(f"  {i+1}/{len(subgraphs)} done")

    # ---- Summary ----
    status_counts = Counter(r["status"] for r in results)
    symint_counts = Counter(len(r["symint_params"]) for r in results)
    shape_patterns = Counter()
    constraint_statuses = Counter(r["constraint_status"] for r in results)
    reifier_dist = Counter(str(r.get("reifier") or "none") for r in results)
    all_symint_values = Counter()

    for r in results:
        if r["first_tensor_shape"]:
            ndim = len(r["first_tensor_shape"])
            pattern = ",".join(['S' if (i < 2 and v > 1) else str(v) for i, v in enumerate(r["first_tensor_shape"])])
            shape_patterns[pattern] += 1
        for s, v in r.get("symint_example_values", {}).items():
            all_symint_values[f"{s}={v}"] += 1

    print(f"\n{'='*60}")
    print(f"Diagnosis: {args.data_dir}")
    print(f"Total subgraphs: {len(results)}")
    print(f"{'='*60}")

    print(f"\n[Status Distribution]")
    for status in ["ready_for_generalization", "needs_reifier", "needs_constraints", "hardcoded", "broken"]:
        count = status_counts.get(status, 0)
        pct = count / max(len(results), 1) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    print(f"\n[SymInt Count Distribution]")
    for c in sorted(symint_counts):
        print(f"  {c} SymInt params: {symint_counts[c]} subgraphs")

    print(f"\n[SymInt Example Values] (all are placeholder values)")
    for val, count in all_symint_values.most_common(10):
        print(f"  {val}: {count} subgraphs")

    print(f"\n[Constraint Status]")
    for s, c in sorted(constraint_statuses.items()):
        print(f"  {s}: {c}")

    print(f"\n[Reifier Distribution]")
    for s, c in sorted(reifier_dist.items()):
        print(f"  {s}: {c}")

    print(f"\n[First Tensor Shape Patterns] (top 20)")
    for pat, c in shape_patterns.most_common(20):
        print(f"  [{pat}]: {c}")

    # Filtered view
    if args.status_filter:
        filtered = [r for r in results if r["status"] == args.status_filter]
        print(f"\n[Filtered: {args.status_filter}] ({len(filtered)} subgraphs)")
        for r in filtered[:20]:
            rel = os.path.relpath(r["path"], args.data_dir)
            line = f"  {rel}"
            if r["symint_params"]:
                line += f" | SymInt: {r['symint_params']}"
            if r["first_tensor_shape"]:
                line += f" | shape: {r['first_tensor_shape']}"
            print(line)
        if len(filtered) > 20:
            print(f"  ... and {len(filtered)-20} more")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
