#!/usr/bin/env python3.10
"""
批量维度泛化脚本
对有 seq_len 的子图执行方案二维度泛化
"""

import os
import re
import shutil
from pathlib import Path

INPUT_DIR = "/ssd1/liangtai-work/lt_submit4.10_sample_100"
OUTPUT_DIR = "/ssd1/liangtai-work/lt_submit4.10_sample_100_dim_gen"

# 统一的 seq_lens 配置（至少 9 组，考虑 max_position=512 的限制）
SEQ_LENS = [64, 128, 256, 512, 64, 128, 256, 512, 128]

def get_seq_len_and_max_pos(wm_content):
    """从 weight_meta.py 提取 seq_len 和 max_position"""
    tensors = {}
    for m in re.finditer(r'class Program_weight_tensor_meta_(\w+):.*?shape\s*=\s*(\[[^\]]+\])', wm_content, re.DOTALL):
        name = m.group(1)
        shape = eval(m.group(2))
        if len(shape) >= 1:
            tensors[name] = shape

    seq_len = None
    max_position = None

    for n, s in tensors.items():
        if n == 's0' and len(s) == 2:
            seq_len = s[1]
        if 'input_ids' in n.lower() and len(s) == 2:
            seq_len = s[1]
        if 'hidden_states' in n.lower() and len(s) == 3:
            seq_len = s[1]
        if 'position_ids' in n.lower() and len(s) == 2:
            max_position = s[1]

    return seq_len, max_position, tensors

def get_hardcoded_dims(model_content):
    """从 model.py 提取硬编码维度"""
    hardcoded = set()
    for m in re.finditer(r'torch\.arange\(0,\s*(\d+)', model_content):
        hardcoded.add(int(m.group(1)))
    for m in re.finditer(r'slice\(0,\s*(\d+)', model_content):
        hardcoded.add(int(m.group(1)))
    for m in re.finditer(r'\.expand\([^,]+,\s*(\d+)\)', model_content):
        hardcoded.add(int(m.group(1)))
    return hardcoded

def adjust_data_field(wm_content, old_seq_len, new_seq_len):
    """调整 data 字段以匹配新的 seq_len"""
    import re

    # 首先更新 s0 (SymInt 参数) 的 data 值
    # s0 通常表示 seq_len
    def replace_s0_data(match):
        return f'{match.group(1)}[{new_seq_len}]'

    s0_pattern = r'(class Program_weight_tensor_meta_s0:.*?data\s*=\s*)\[(\d+)\]'
    wm_content = re.sub(s0_pattern, replace_s0_data, wm_content, flags=re.DOTALL)

    # 分割成各个 class 块
    class_pattern = r'(class Program_weight_tensor_meta_\w+:(?:\n\t[^\n]+)+)'
    classes = re.findall(class_pattern, wm_content)

    result = wm_content

    for class_block in classes:
        # 提取 class 名称
        class_name_match = re.search(r'class Program_weight_tensor_meta_(\w+):', class_block)
        if not class_name_match:
            continue

        # 检查 shape
        shape_match = re.search(r'shape\s*=\s*(\[[^\]]+\])', class_block)
        if not shape_match:
            continue

        shape_str = shape_match.group(1)

        # 只处理 [1, old_seq_len] 的 tensor
        if shape_str != f'[1, {old_seq_len}]':
            continue

        # 提取 data
        data_match = re.search(r'data\s*=\s*\[([^\]]*)\]', class_block)
        if not data_match:
            continue

        data_str = data_match.group(1)
        if not data_str.strip():
            continue

        try:
            # 解析数据
            old_data = []
            for x in data_str.split(','):
                x = x.strip()
                if not x:
                    continue
                if '.' in x or 'e' in x.lower():
                    old_data.append(float(x))
                else:
                    old_data.append(int(x))

            # 调整 data 长度
            if new_seq_len > len(old_data):
                new_data = old_data * (new_seq_len // len(old_data)) + old_data[:new_seq_len % len(old_data)]
            else:
                new_data = old_data[:new_seq_len]

            # 格式化输出
            if all(isinstance(x, int) for x in new_data):
                new_data_str = str(new_data)
            else:
                new_data_str = '[' + ', '.join(f'{x:.6f}' if isinstance(x, float) else str(x) for x in new_data) + ']'

            # 替换 data
            old_data_line = f'data = [{data_str}]'
            new_data_line = f'data = {new_data_str}'
            result = result.replace(old_data_line, new_data_line)
        except ValueError:
            pass

    return result


def generate_variants(sg_path, sg_name, output_dir, seq_lens, old_seq_len, hardcoded):
    """生成维度变体"""
    wm_path = os.path.join(sg_path, "weight_meta.py")
    model_path = os.path.join(sg_path, "model.py")

    with open(wm_path) as f:
        wm_content_orig = f.read()
    with open(model_path) as f:
        model_content_orig = f.read()

    generated = []
    for idx, new_seq_len in enumerate(seq_lens):
        # 输出路径：output_dir/idx/subgraph_name/
        out_path = os.path.join(output_dir, str(idx), sg_name)
        os.makedirs(out_path, exist_ok=True)

        # 复制整个子图目录
        shutil.copytree(Path(sg_path), Path(out_path), dirs_exist_ok=True)

        # 修改 weight_meta.py
        wm_out = os.path.join(out_path, "weight_meta.py")
        wm_content = wm_content_orig

        # 先调整 data 字段（基于原始 shape）
        wm_content = adjust_data_field(wm_content, old_seq_len, new_seq_len)

        # 只替换与 seq_len 相关的 tensor 的 shape
        # 需要替换的 tensor 名称模式
        seq_len_patterns = [
            f'shape = [1, {old_seq_len}]',  # input_ids, attention_mask 等
        ]

        for pattern in seq_len_patterns:
            if old_seq_len <= new_seq_len:
                wm_content = wm_content.replace(pattern, pattern.replace(str(old_seq_len), str(new_seq_len)))

        with open(wm_out, 'w') as f:
            f.write(wm_content)

        # 修改 model.py
        model_out = os.path.join(out_path, "model.py")
        model_content = model_content_orig
        for dim in hardcoded:
            model_content = model_content.replace(f'torch.arange(0, {dim},', f'torch.arange(0, {new_seq_len},')
            model_content = model_content.replace(f'slice(0, {dim},', f'slice(0, {new_seq_len},')
            model_content = model_content.replace(f'.expand(1, {dim})', f'.expand(1, {new_seq_len})')
        with open(model_out, 'w') as f:
            f.write(model_content)

        generated.append({"idx": idx, "seq_len": new_seq_len, "path": out_path})

    return generated

def main():
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"seq_lens 配置: {SEQ_LENS}")
    print()

    # 找所有有 seq_len 的子图
    subgraphs = []
    for sg_name in sorted(os.listdir(INPUT_DIR)):
        if not sg_name.startswith('subgraph_'):
            continue
        sg_path = os.path.join(INPUT_DIR, sg_name)
        wm_path = os.path.join(sg_path, 'weight_meta.py')

        if not os.path.exists(wm_path):
            continue

        with open(wm_path) as f:
            wm_content = f.read()

        seq_len, max_position, tensors = get_seq_len_and_max_pos(wm_content)

        if seq_len is not None:
            subgraphs.append({
                'name': sg_name,
                'path': sg_path,
                'seq_len': seq_len,
                'max_position': max_position,
                'tensors': tensors
            })

    print(f"找到 {len(subgraphs)} 个有 seq_len 的子图")
    print()

    # 处理每个子图
    for i, sg in enumerate(subgraphs):
        print(f"[{i+1}/{len(subgraphs)}] 处理 {sg['name']}...")
        print(f"  seq_len={sg['seq_len']}, max_position={sg['max_position']}")

        # 检查 model.py 中的硬编码
        model_path = os.path.join(sg['path'], 'model.py')
        with open(model_path) as f:
            model_content = f.read()
        hardcoded = get_hardcoded_dims(model_content)

        if hardcoded:
            print(f"  硬编码维度: {hardcoded}")

        # 生成变体
        generated = generate_variants(
            sg['path'], sg['name'], OUTPUT_DIR, SEQ_LENS,
            sg['seq_len'], hardcoded
        )
        print(f"  生成 {len(generated)} 个变体")

    print()
    print(f"完成！共处理 {len(subgraphs)} 个子图，每个生成 {len(SEQ_LENS)} 个变体")

if __name__ == "__main__":
    main()