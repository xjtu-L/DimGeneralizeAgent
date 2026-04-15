# 维度泛化进度

> Agent 启动后先读本文件。只看"当前任务"和"各目录进度"两节即可开始工作。

## 当前任务

**使用方案二处理 lt_submit4.10_sample_100 目录**

方案二流程：问LLM要配置 → 改输入维度 → 改model.py → 验证

**关键约束**：
- **只改输入 tensor 的维度，不改权重的维度**
- 只修改 `L_input_ids_` 的 shape

详见 [docs/plan_b_generalize.md](docs/plan_b_generalize.md)

## 各目录进度

处理顺序: 4.10 → 4.7 → 4.9 → 4.13 → 4.3（从小到大）

| 目录 | 子图数 | 已analyze | 已generate | 已verify | 状态 |
|------|--------|-----------|------------|----------|------|
| lt_submit4.10 | 397 | 0 | 0 | 0 | 未开始 |
| lt_submit4.7 | 484 | 0 | 0 | 0 | 未开始 |
| lt_submit4.9 | 2064 | 0 | 0 | 0 | 未开始 |
| lt_submit4.13 | 2687 | 0 | 0 | 0 | 未开始 |
| lt_submit4.3 | 3777 | 0 | 0 | 0 | 未开始 |
| **合计** | **9409** | **0** | **0** | **0** | |

## 方案二命令速查

```bash
# 设置环境
source /ssd1/liangtai-work/DimGeneralizeAgent/scripts/setup_env.sh

# 1. 分析子图，获取维度信息
python3.10 scripts/plan_b_generalize.py analyze <subgraph_path>

# 2. 问 LLM 推理维度配置（至少9组，输出 prompt）
python3.10 scripts/plan_b_generalize.py ask-llm <subgraph_path>

# 3. 生成变体（用 LLM 返回的值）
python3.10 scripts/plan_b_generalize.py generate <subgraph_path> \n    --output-dir <output_dir> \n    --seq-lens <LLM返回的值，逗号分隔>

# 4. 验证变体
python3.10 scripts/plan_b_generalize.py verify <output_dir>
```

**注意**：变体数量由 LLM 决定，至少 9 组，上不封顶。

## 损坏子图（跳过）

| 目录 | 数量 |
|------|------|
| lt_submit4.7 | 2 |
| lt_submit4.9 | 5 |
| lt_submit4.3 | 21 |
| lt_submit4.10 | 1 |
| lt_submit4.13 | 14 |
