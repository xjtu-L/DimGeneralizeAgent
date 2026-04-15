# 维度泛化方案二：直接改模型

## 核心思路

```
问LLM要配置 → 改输入维度 → 改model.py → 跑模型验证
```

**关键约束**：
- **只改输入 tensor 的维度，不改权重的维度**
- 只修改 `L_input_ids_` 的 shape
- `position_ids` 等其他输入的 shape 保持不变
- 变体数量由 LLM 决定，至少 9 组，上不封顶

## 详细步骤

### Step 1: 分析子图

识别关键维度信息：
- `input_ids` 的 shape → 确定 seq_len
- `position_embeddings` 的 shape → 确定最大位置
- 其他 tensor 的 shape → 确定维度约束

```bash
python3.10 scripts/plan_b_generalize.py analyze <subgraph_path>
```

### Step 2: 问 LLM 要配置

```bash
python3.10 scripts/plan_b_generalize.py ask-llm <subgraph_path>
```

**输出**：
- 构造好的 LLM prompt
- 包含模型信息、约束条件

**LLM 决策内容**：
- 生成多少组配置（至少 9 组）
- 每组的 seq_len 值
- 考虑短文本、中等长度、长文本等不同场景

**示例 LLM 输出**：
```json
[64, 128, 256, 512, 64, 128, 256, 512, 128, 256, 512, 1024]
```

### Step 3: 生成变体

```bash
python3.10 scripts/plan_b_generalize.py generate <subgraph_path> \n    --output-dir <output_dir> \n    --seq-lens 64,128,256,512,64,128,256,512,128,256,512,1024
```

**修改内容**：
- `weight_meta.py`: `input_ids` shape 改为 `[1, new_seq_len]`
- `model.py`: 修复硬编码维度值

**关键**：`position_ids` shape **保持不变**，model.py 通过 slice 取前 seq_len 个

### Step 4: 验证

```bash
python3.10 scripts/plan_b_generalize.py verify <output_dir>
```

自动检测变体数量，验证所有变体。

---

## 测试结果

**测试样本**: `finiteautomata_beto-sentiment-analysis/subgraph_0`

| 变体 | seq_len | 结果 |
|------|---------|------|
| 0 | 64 | ✓ OK |
| 1 | 128 | ✓ OK |
| 2 | 256 | ✓ OK |
| 3 | 512 | ✓ OK |
| 4 | 64 | ✓ OK |
| 5 | 128 | ✓ OK |
| 6 | 256 | ✓ OK |
| 7 | 512 | ✓ OK |
| 8 | 128 | ✓ OK |

**通过率**: 9/9 = 100%

---

## 与方案一对比

| | 方案一（符号化） | 方案二（直接改） |
|---|---|---|
| 流程 | 符号化→具体化→泛化 | 问配置→改meta→改model |
| 复杂度 | 高 | **低** |
| 通过率 | 1/9 | **9/9** |
| 关键修复 | 需更新 position_ids 数据 | 需修复 model.py 硬编码 |
| 适用场景 | 通用框架 | **生产环境快速处理** |

**结论**: 方案二更实用，后续工作统一使用方案二。

---

## 实现脚本

见 `scripts/plan_b_generalize.py`（待创建）

---

## 注意事项

1. **position_ids 不要改 shape**：它存储的是最大位置的索引，model.py 通过 slice 取前 seq_len 个
2. **model.py 硬编码**：需要扫描并修复所有硬编码的维度值
3. **维度约束**：seq_len 不能超过 position_embeddings 的最大位置