#!/usr/bin/env python3
"""
LLM 具体化工具：将符号化的维度模式交给 Agent 推理，生成具体维度值。

流程：
1. gen-constraints 输出符号模式信息
2. Agent（Claude/GLM等）根据符号模式推理出推荐的维度值
3. reify-by-llm 将 Agent 推理的值写入配置

用法：
    # 1. 生成符号化信息
    python3.10 llm_reifier.py extract-symbols <subgraph_path>

    # 2. Agent 推理（见下方 prompt 模板）

    # 3. 写入维度值
    python3.10 llm_reifier.py apply-values <subgraph_path> --values "S0=64,128,256 S1=1,1,1"
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, "/ssd1/liangtai-work/GraphNet")

from graph_net.dynamic_dim_constraints import DynamicDimConstraints


def extract_symbols(sg_path):
    """提取符号化信息，输出给 Agent 推理"""
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")

    if not os.path.exists(cstr_path):
        print(f"ERROR: No constraints file in {sg_path}")
        print("Run gen-constraints first.")
        return None

    try:
        cstr = DynamicDimConstraints.unserialize_from_py_file(cstr_path)
    except Exception as e:
        print(f"ERROR: Failed to parse constraints: {e}")
        return None

    if len(cstr.symbols) == 0:
        print(f"ERROR: No symbols found in {sg_path}")
        return None

    # 提取信息
    symbols = [str(s) for s in cstr.symbols]
    example_values = {str(k): v for k, v in cstr.symbol2example_value.items()}
    symbolic_shapes = cstr.serialize_symbolic_input_shapes_to_str()

    # 分析输入 tensor 信息
    input_shapes = []
    for shape, name in cstr.input_shapes:
        shape_str = [str(d) for d in shape]
        input_shapes.append({"name": name, "shape": shape_str})

    result = {
        "path": sg_path,
        "symbols": symbols,
        "example_values": example_values,
        "symbolic_shapes": symbolic_shapes,
        "input_shapes": input_shapes,
    }

    # 输出 JSON 和人类可读格式
    print("\n" + "=" * 60)
    print("符号化信息（供 Agent 推理）")
    print("=" * 60)

    print(f"\n## 符号: {', '.join(symbols)}")
    print(f"\n## 示例值: {example_values}")
    print(f"\n## 符号化形状: {symbolic_shapes}")
    print(f"\n## 输入 Tensor:")
    for inp in input_shapes[:5]:  # 最多显示5个
        print(f"  - {inp['name']}: {inp['shape']}")
    if len(input_shapes) > 5:
        print(f"  - ... 还有 {len(input_shapes) - 5} 个")

    print("\n" + "=" * 60)
    print("JSON 格式：")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


def apply_values(sg_path, values_str):
    """将 Agent 推理的维度值写入配置"""
    cstr_path = os.path.join(sg_path, "input_tensor_constraints.py")

    if not os.path.exists(cstr_path):
        print(f"ERROR: No constraints file in {sg_path}")
        return False

    try:
        cstr = DynamicDimConstraints.unserialize_from_py_file(cstr_path)
    except Exception as e:
        print(f"ERROR: Failed to parse constraints: {e}")
        return False

    # 解析 Agent 输入的值
    # 格式: "S0=64,128,256 S1=1,1,1" 或 "S0:64,128,256;S1:1,1,1"
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
        if sym not in [str(s) for s in cstr.symbols]:
            print(f"WARNING: Unknown symbol: {sym}")

    # 检查各组长度是否一致
    lengths = [len(v) for v in values.values()]
    if len(set(lengths)) > 1:
        print(f"ERROR: Inconsistent value counts: {dict((k, len(v)) for k, v in values.items())}")
        print("All symbols must have the same number of values (e.g., 9 values each)")
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
    return True


def generate_variants(sg_path):
    """根据 llm_reified_values.json 生成维度变体"""
    # 读取 LLM 推理的值
    values_path = os.path.join(sg_path, "llm_reified_values.json")
    if not os.path.exists(values_path):
        print(f"ERROR: No LLM values file in {sg_path}")
        print("Run apply-values first.")
        return False

    with open(values_path) as f:
        llm_values = json.load(f)

    symbols = llm_values["symbols"]
    values = llm_values["values"]
    num_variants = llm_values["num_variants"]

    # 生成变体列表
    variants = []
    for i in range(num_variants):
        variant = {}
        for sym in symbols:
            variant[sym] = values[sym][i]
        variants.append(variant)

    print(f"\n生成的 {num_variants} 个维度变体:")
    for i, v in enumerate(variants):
        print(f"  [{i}] {v}")

    # 保存变体列表
    variants_path = os.path.join(sg_path, "llm_variants.json")
    with open(variants_path, "w") as f:
        json.dump(variants, f, indent=2)

    print(f"\nOK: Variants saved to {variants_path}")
    return variants


def main():
    parser = argparse.ArgumentParser(description="LLM 具体化工具")
    subparsers = parser.add_subparsers(dest="command")

    # extract-symbols
    p = subparsers.add_parser("extract-symbols", help="提取符号化信息供 Agent 推理")
    p.add_argument("sg_path")

    # apply-values
    p = subparsers.add_parser("apply-values", help="写入 Agent 推理的维度值")
    p.add_argument("sg_path")
    p.add_argument("--values", "-v", required=True, help='格式: "S0=64,128,256 S1=1,1,1"')

    # generate-variants
    p = subparsers.add_parser("generate-variants", help="生成维度变体列表")
    p.add_argument("sg_path")

    args = parser.parse_args()

    if args.command == "extract-symbols":
        extract_symbols(args.sg_path)
    elif args.command == "apply-values":
        apply_values(args.sg_path, args.values)
    elif args.command == "generate-variants":
        generate_variants(args.sg_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()