# 维度泛化进度

> Agent 启动后先读本文件。只看"当前任务"和"各目录进度"两节即可开始工作。

## 当前任务

**lt_submit4.7: gen-constraints (还有 415 个子图需要生成约束)**

## 各目录进度

处理顺序: 4.7 → 4.9 → 4.3（从小到大）

| 目录 | 子图数 | 需gen-constraints | 需assign-reifier | 已generalize | 状态 |
|------|--------|-------------------|------------------|-------------|------|
| lt_submit4.7 | 484 | 415 | 0 | 0 | 未开始 |
| lt_submit4.9 | 2064 | 1304 | 0 | 0 | 未开始 |
| lt_submit4.3 | 3777 | 1947 | 0 | 0 | 未开始 |
| **合计** | **6325** | **3666** | **0** | **0** | |

## 硬编码子图（需先 symbolize 再走后续流程）

| 目录 | 硬编码数 | symbolize 成功 | symbolize 失败 | 状态 |
|------|---------|---------------|---------------|------|
| lt_submit4.7 | 67 | 0 | 0 | 未开始 |
| lt_submit4.9 | 755 | 0 | 0 | 未开始 |
| lt_submit4.3 | 1809 | 0 | 0 | 未开始 |

## 损坏子图（跳过）

| 目录 | 数量 |
|------|------|
| lt_submit4.7 | 2 |
| lt_submit4.9 | 5 |
| lt_submit4.3 | 21 |

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
