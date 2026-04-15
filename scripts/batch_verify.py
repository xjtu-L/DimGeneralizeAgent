#!/usr/bin/env python3.10
"""
批量验证维度变体
"""

import os
import subprocess
import json
from datetime import datetime

OUTPUT_DIR = "/ssd1/liangtai-work/lt_submit4.10_sample_100_dim_gen"

def verify_variant(model_path):
    """验证单个变体"""
    try:
        result = subprocess.run(
            ["python3.10", "-m", "graph_net.torch.run_model", "--model-path", model_path],
            capture_output=True, text=True, timeout=60,
            cwd="/ssd1/liangtai-work/GraphNet"
        )
        success = result.returncode == 0
        error_msg = result.stderr[-500:] if not success else None
        return success, error_msg
    except Exception as e:
        return False, str(e)

def main():
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"开始时间: {datetime.now()}")
    print()

    # 找所有变体目录
    variant_dirs = []
    for idx in range(100):  # 最多 100 个变体
        variant_path = os.path.join(OUTPUT_DIR, str(idx))
        if os.path.exists(variant_path):
            variant_dirs.append((idx, variant_path))
        else:
            break

    print(f"找到 {len(variant_dirs)} 个变体目录")
    print()

    # 统计结果
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "details": {}
    }

    # 验证每个变体
    for idx, variant_dir in variant_dirs:
        print(f"=== 变体 {idx} (seq_len={[(64,128,256,512,64,128,256,512,128)[idx]]}) ===")

        # 找变体目录中的所有子图
        for sg_name in sorted(os.listdir(variant_dir)):
            if not sg_name.startswith('subgraph_'):
                continue

            sg_path = os.path.join(variant_dir, sg_name)
            model_path = os.path.join(sg_path, "model.py")

            if not os.path.exists(model_path):
                print(f"  {sg_name}: SKIP (no model.py)")
                continue

            results["total"] += 1
            success, error = verify_variant(sg_path)

            if success:
                results["passed"] += 1
                print(f"  {sg_name}: OK")
            else:
                results["failed"] += 1
                print(f"  {sg_name}: FAILED")
                if error:
                    print(f"    Error: {error[:200]}")

            # 记录详情
            if sg_name not in results["details"]:
                results["details"][sg_name] = []
            results["details"][sg_name].append({
                "variant": idx,
                "success": success
            })

        print()

    # 打印统计
    print("=" * 60)
    print("验证结果统计")
    print("=" * 60)
    print(f"总测试数: {results['total']}")
    print(f"通过: {results['passed']}")
    print(f"失败: {results['failed']}")
    print(f"通过率: {results['passed']/results['total']*100:.1f}%")
    print()

    # 按子图统计
    print("各子图通过情况:")
    for sg_name, variants in sorted(results["details"].items()):
        passed = sum(1 for v in variants if v["success"])
        total = len(variants)
        print(f"  {sg_name}: {passed}/{total}")

    print()
    print(f"结束时间: {datetime.now()}")

    # 保存结果
    result_file = os.path.join(OUTPUT_DIR, "verify_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存到: {result_file}")

if __name__ == "__main__":
    main()