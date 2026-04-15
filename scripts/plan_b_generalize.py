#!/usr/bin/env python3
"""
方案二：直接改模型实现维度泛化

流程：问LLM要配置 → 改meta → 改model.py → 验证

用法：
    # 1. 分析子图，获取维度信息
    python3.10 plan_b_generalize.py analyze <subgraph_path>

    # 2. 问 LLM 推理维度配置（至少9组）
    python3.10 plan_b_generalize.py ask-llm <subgraph_path>

    # 3. 生成变体
    python3.10 plan_b_generalize.py generate <subgraph_path> --output-dir <output_dir> --seq-lens 64,128,256,512

    # 4. 验证变体
    python3.10 plan_b_generalize.py verify <output_dir>
"""

import argparse
import os
import sys
import re
import json
import shutil
from pathlib import Path

sys.path.insert(0, "/ssd1/liangtai-work/GraphNet")


def analyze_subgraph(sg_path):
    """分析子图，提取维度信息"""
    wm_path = os.path.join(sg_path, "weight_meta.py")
    model_path = os.path.join(sg_path, "model.py")

    if not os.path.exists(wm_path):
        print(f"ERROR: No weight_meta.py in {sg_path}")
        return None

    with open(wm_path) as f:
        wm_content = f.read()

    with open(model_path) as f:
        model_content = f.read()

    # 提取关键 tensor 信息
    tensors = {}
    for match in re.finditer(r'class Program_weight_tensor_meta_(\w+):.*?shape\s*=\s*(\[[^\]]+\])', wm_content, re.DOTALL):
        name = match.group(1)
        shape = eval(match.group(2))
        tensors[name] = shape

    # 找 seq_len（从 input_ids）
    seq_len = None
    max_position = None

    for name, shape in tensors.items():
        if 'input_ids' in name.lower():
            if len(shape) >= 2:
                seq_len = shape[1]
        if 'position_embeddings' in name.lower():
            if len(shape) >= 1:
                max_position = shape[0]

    # 找 model.py 中的硬编码维度
    hardcoded_dims = set()
    for m in re.finditer(r'torch\.arange\(0,\s*(\d+)', model_content):
        hardcoded_dims.add(int(m.group(1)))
    for m in re.finditer(r'slice\(0,\s*(\d+)', model_content):
        hardcoded_dims.add(int(m.group(1)))
    for m in re.finditer(r'\.expand\([^,]+,\s*(\d+)\)', model_content):
        hardcoded_dims.add(int(m.group(1)))

    result = {
        "path": sg_path,
        "seq_len": seq_len,
        "max_position": max_position,
        "hardcoded_dims": list(hardcoded_dims),
        "tensors": {k: v for k, v in tensors.items() if len(v) > 0},
    }

    print("\n" + "=" * 60)
    print("子图维度分析")
    print("=" * 60)
    print(f"\n路径: {sg_path}")
    print(f"当前 seq_len: {seq_len}")
    print(f"最大位置: {max_position}")
    print(f"硬编码维度: {hardcoded_dims}")

    print("\n关键 Tensor:")
    for name, shape in tensors.items():
        if len(shape) > 0:
            print(f"  {name}: {shape}")

    # 给 LLM 的提示
    print("\n" + "-" * 60)
    print("请 LLM 推理 9 组 seq_len 值：")
    if max_position:
        print(f"  约束: seq_len <= {max_position}")
    print("  推荐格式: 64,128,256,512,64,128,256,512,128")
    print("-" * 60)

    return result


def ask_llm_for_config(sg_path):
    """问 LLM 推理维度配置"""
    # 先分析子图
    info = analyze_subgraph(sg_path)
    if info is None:
        return None

    # 构造 LLM prompt
    prompt = f"""你是一个深度学习模型维度配置专家。请根据以下模型信息，推理出合适的 seq_len 配置。

## 模型信息
- 当前 seq_len: {info['seq_len']}
- 最大位置 (max_position): {info['max_position']}
- 硬编码维度: {info['hardcoded_dims']}

## 关键 Tensor 形状
"""
    for name, shape in info['tensors'].items():
        prompt += f"- {name}: {shape}\n"

    prompt += f"""
## 约束条件
- seq_len 不能超过 max_position ({info['max_position']})
- 至少生成 9 组配置，可以更多
- 考虑模型的实际使用场景，选择有代表性的 seq_len 值

## 输出要求
请直接输出 JSON 数组，包含推荐的 seq_len 值。例如：
[64, 128, 256, 512, 64, 128, 256, 512, 128, 256]

注意：
1. 不需要解释，直接输出 JSON 数组
2. 至少 9 个值
3. 值应该覆盖不同的使用场景（短文本、中等长度、长文本）
4. 可以有重复值（表示某些长度更常用）
"""

    print("\n" + "=" * 60)
    print("LLM Prompt:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print("\n请将上述 prompt 发送给 LLM，获取 seq_lens 配置。")
    print("然后使用以下命令生成变体：")
    print(f"  python3.10 plan_b_generalize.py generate {sg_path} -o <output_dir> --seq-lens <LLM返回的值>")

    return {"prompt": prompt, "info": info}


def generate_variants(sg_path, output_dir, seq_lens):
    """生成维度变体"""
    if len(seq_lens) < 9:
        print(f"WARNING: seq_lens 只有 {len(seq_lens)} 个，建议至少 9 个")
    wm_path = os.path.join(sg_path, "weight_meta.py")
    model_path = os.path.join(sg_path, "model.py")

    with open(wm_path) as f:
        wm_content_orig = f.read()
    with open(model_path) as f:
        model_content_orig = f.read()

    # 找原始 seq_len (从 L_input_ids_)
    m = re.search(r'class Program_weight_tensor_meta_L_input_ids_:.*?shape\s*=\s*\[1,\s*(\d+)\]', wm_content_orig, re.DOTALL)
    old_seq_len = int(m.group(1)) if m else None

    if old_seq_len is None:
        print("WARNING: 未找到 L_input_ids_，跳过")
        return []

    print(f"\n原始 seq_len: {old_seq_len}")
    print(f"目标 seq_lens: {seq_lens}")

    # 获取相对路径
    parent_dir = os.path.dirname(os.path.dirname(sg_path))
    rel_path = os.path.relpath(sg_path, parent_dir)

    generated = []
    for idx, new_seq_len in enumerate(seq_lens):
        out_path = os.path.join(output_dir, str(idx), rel_path)
        os.makedirs(out_path, exist_ok=True)
        shutil.copytree(Path(sg_path), Path(out_path), dirs_exist_ok=True)

        # 修改 weight_meta.py - 只改输入 tensor (L_input_ids_) 的 shape
        wm_out = os.path.join(out_path, "weight_meta.py")
        wm_content = wm_content_orig
        # 精确匹配 L_input_ids_ 类的 shape，不改权重
        pattern = r'(class Program_weight_tensor_meta_L_input_ids_:.*?shape\s*=\s*)\[1,\s*\d+\]'
        wm_content = re.sub(pattern, rf'\1[1, {new_seq_len}]', wm_content, flags=re.DOTALL)
        with open(wm_out, 'w') as f:
            f.write(wm_content)

        # 修改 model.py - 只替换与 old_seq_len 相等的硬编码值
        model_out = os.path.join(out_path, "model.py")
        model_content = model_content_orig
        # 只替换等于 old_seq_len 的硬编码维度
        model_content = model_content.replace(f'torch.arange(0, {old_seq_len},', f'torch.arange(0, {new_seq_len},')
        model_content = model_content.replace(f'slice(0, {old_seq_len},', f'slice(0, {new_seq_len},')
        model_content = model_content.replace(f'.expand(1, {old_seq_len})', f'.expand(1, {new_seq_len})')
        model_content = model_content.replace(f'.expand({old_seq_len})', f'.expand({new_seq_len})')
        with open(model_out, 'w') as f:
            f.write(model_content)

        generated.append({"idx": idx, "seq_len": new_seq_len, "path": out_path})
        print(f"  [{idx}] seq_len={new_seq_len}")

    print(f"\n生成 {len(generated)} 个变体")
    return generated


def verify_variants(output_dir):
    """验证所有变体"""
    import subprocess

    results = []

    # 自动检测变体目录数量
    variant_dirs = []
    for idx in range(1000):  # 最多支持 1000 个变体
        variant_path = os.path.join(output_dir, str(idx))
        if os.path.exists(variant_path):
            variant_dirs.append(variant_path)
        else:
            break

    if not variant_dirs:
        print(f"ERROR: 在 {output_dir} 中未找到变体目录")
        return results

    print(f"找到 {len(variant_dirs)} 个变体目录")

    for idx, variant_dir in enumerate(variant_dirs):
        # 找变体目录中的 model.py
        for root, dirs, files in os.walk(variant_dir):
            if "model.py" in files:
                variant_path = root
                break
        else:
            results.append({"idx": idx, "success": False, "path": variant_dir, "error": "model.py not found"})
            print(f"  [{idx}] ERROR: model.py not found")
            continue

        try:
            result = subprocess.run(
                ["python3.10", "-m", "graph_net.torch.run_model", "--model-path", variant_path],
                capture_output=True, text=True, timeout=60,
                cwd="/ssd1/liangtai-work/GraphNet"
            )
            success = result.returncode == 0
            error_msg = result.stderr[-500:] if not success else None
        except Exception as e:
            success = False
            error_msg = str(e)

        results.append({"idx": idx, "success": success, "path": variant_path})
        status = "OK" if success else "ERROR"
        print(f"  [{idx}] {status}")

    passed = sum(1 for r in results if r["success"])
    print(f"\n验证结果: {passed}/{len(results)} 通过")
    return results


def main():
    parser = argparse.ArgumentParser(description="方案二：直接改模型实现维度泛化")
    subparsers = parser.add_subparsers(dest="command")

    # analyze
    p = subparsers.add_parser("analyze", help="分析子图，提取维度信息")
    p.add_argument("sg_path")

    # ask-llm
    p = subparsers.add_parser("ask-llm", help="问 LLM 推理维度配置（至少9组）")
    p.add_argument("sg_path")

    # generate
    p = subparsers.add_parser("generate", help="生成维度变体")
    p.add_argument("sg_path")
    p.add_argument("--output-dir", "-o", required=True)
    p.add_argument("--seq-lens", "-s", required=True,
                   help="seq_len值列表，逗号分隔，至少9个")

    # verify
    p = subparsers.add_parser("verify", help="验证变体")
    p.add_argument("output_dir")

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_subgraph(args.sg_path)
    elif args.command == "ask-llm":
        ask_llm_for_config(args.sg_path)
    elif args.command == "generate":
        seq_lens = [int(x.strip()) for x in args.seq_lens.split(",")]
        generate_variants(args.sg_path, args.output_dir, seq_lens)
    elif args.command == "verify":
        verify_variants(args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()