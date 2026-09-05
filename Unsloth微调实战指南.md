# Unsloth 微调实战指南

> **运行方式**：全部使用云端 GPU 运行，推荐 Google Colab 免费 T4（≈15GB VRAM）或 Kaggle 免费额度（每周 30h + 2×T4）。无需本地 GPU，代码零改动即可运行。
>
> **学习目标**：掌握 Unsloth 微调框架，能够独立完成 Qwen2.5 模型的微调与部署
>
> **适用硬件**：Google Colab 免费 T4 GPU（≈15GB）；网络不稳备选 Kaggle 免费额度
>
> **前置知识**：已掌握 LoRA/QLoRA 基础理论（参见 `LoRA与QLoRA微调大语言模型完整指南.md`）

---

## 运行方式：云端 GPU 运行指南

> **一条原则**：所有微调任务均在云端 GPU 运行，推荐 Google Colab 免费 T4（≈15GB VRAM）；网络不稳时备选 Kaggle 免费额度（每周 30h + 2×T4）。无需本地 GPU，代码零改动。

### 1. 云端平台选择

| 平台 | GPU 配置 | 免费额度 | 优势 |
|------|----------|----------|------|
| **Google Colab** | T4（≈15GB） | 每天免费使用（有时限） | 官方 Notebook 直接运行，生态完善 |
| **Kaggle** | 2×T4（≈30GB） | 每周 30 小时 | 显存更大，适合 14B+ 模型 |
| **其他云 GPU** | A100/H100（按需） | 付费 | 适合大规模训练 |

**推荐路径**：入门用 Colab T4 → 大模型用 Kaggle 2×T4 → 生产用付费云 GPU。

### 2. 模型显存需求对照（Colab T4 ≈15GB）

| 模型参数 | 最低显存 | Colab T4 能跑 | 说明 |
|----------|----------|----------------|------|
| 3B | 3.5 GB | ✅ 轻松 | 入门首选 |
| 7B | 5 GB | ✅ 轻松 | 核心学习内容 |
| 8B | 6 GB | ✅ 轻松 | Llama 3.1 等 |
| 9B | 6.5 GB | ✅ 轻松 | - |
| 14B | 8.5 GB | ✅ 可跑 | Phi-4、Qwen3 14B 等 |
| 20B | 10 GB+ | ⚠️ 勉强（QLoRA） | gpt-oss 20B 等 |
| 27B | 22 GB | ❌ 需 Kaggle 2×T4 | 大模型 |

### 3. 云端运行推荐配置

| 模型 | 平台 | batch_size | grad_accum | max_seq_len | 说明 |
|------|------|------------|------------|-------------|------|
| 3B QLoRA | Colab T4 | 4 | 2 | 2048 | 从容运行 |
| 7B QLoRA | Colab T4 | 2 | 4 | 2048 | 标准配置 |
| 8B QLoRA | Colab T4 | 2 | 4 | 2048 | Llama 3.1 等 |
| 14B QLoRA | Kaggle 2×T4 | 2 | 4 | 2048 | 需更大显存 |
| 7B LoRA(16-bit) | Kaggle 2×T4 | 1 | 8 | 2048 | 19GB 显存需求 |

### 4. 云端运行要点

1. 本文「Free Notebooks」那节的链接**点开就是 Colab 版**，运行时菜单选 **T4 GPU**（免费档）
2. 免费额度有时限、易断线：**数据集和脚本放 Google Drive**，断线重连后从 checkpoint 续训
3. Kaggle 大陆网络不稳时备选 **Colab**，或使用 VPN
4. 产出物（adapter / GGUF）下载到本地计算机，后续 Ollama 部署、评测在本地计算机进行
5. 大模型（27B+）需升级到付费云 GPU（A100/H100）

---

## Unsloth 简介

**Unsloth** 是一个开源的 LLM 微调和推理加速工具，通过自定义 Triton 内核优化，在 HuggingFace 生态之上提供极致的训练和推理性能。

### 核心定位
- **本质**：建立在 HuggingFace 之上的加速器（不是替代品）
- **形式**：本地桌面应用（Unsloth Desktop）+ Web UI（Unsloth Studio）
- **能力**：运行、微调、部署 500+ AI 模型

### 与 HuggingFace 的关系

| 维度 | HuggingFace | Unsloth |
|------|-------------|---------|
| 定位 | 完整 AI 生态（模型库+工具链+社区） | 专注微调/推理的加速工具 |
| 底层 | PyTorch 标准实现 | 自定义 Triton 内核优化 |
| 性能 | 基准速度 | 快 2x，省 70% VRAM |
| API 复杂度 | 需组合多个库（Transformers+PEFT+TRL） | 封装好的 `FastLanguageModel`，更简洁 |

### 代码对比

```python
# HuggingFace 方式（需要多个库）
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", quantization_config=bnb_config)
lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)

# Unsloth 方式（一行搞定）
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-7B-bnb-4bit")
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)
```

### 核心优势

- **2x 更快的训练速度**：通过自定义 Triton 内核优化
- **70% 更少的 VRAM 使用**：支持 4-bit 量化和梯度检查点
- **无精度损失**：动态 4-bit 量化技术
- **支持 500+ 模型**：包括 Qwen、Llama、Gemma、DeepSeek 等

- **2x 更快的训练速度**：通过自定义 Triton 内核优化
- **70% 更少的 VRAM 使用**：支持 4-bit 量化和梯度检查点
- **无精度损失**：动态 4-bit 量化技术
- **支持 500+ 模型**：包括 Qwen、Llama、Gemma、DeepSeek 等

### Unsloth 与原生 HuggingFace 的代码对比

```python
# 原生 HuggingFace 方式（已学过）
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Unsloth 方式（更简洁高效）
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(model_name="unsloth/Qwen2.5-7B")
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)
```

---

## 第一阶段：云端环境配置

### 1. 云端环境要求

| 组件 | 要求 |
|------|------|
| 平台 | Google Colab / Kaggle |
| Python | 3.9+（云端预装） |
| CUDA | 11.8+（云端预装） |
| GPU | T4（≈15GB）或更高 |

### 2. Colab 环境配置

#### 步骤1：选择 GPU 运行时

```python
# 在 Colab 中运行，确认 GPU 可用
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

#### 步骤2：安装 Unsloth

```bash
# 在 Colab 中运行
!pip install unsloth
!pip install --upgrade --no-cache-dir torch trl peft accelerate
```

#### 步骤3：挂载 Google Drive（推荐）

```python
# 挂载 Google Drive，用于持久化数据和检查点
from google.colab import drive
drive.mount('/content/drive')

# 创建工作目录
!mkdir -p /content/drive/MyDrive/unsloth-project
```

### 3. Kaggle 环境配置

#### 步骤1：启用 GPU

```python
# 在 Kaggle Notebook 中运行
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

#### 步骤2：安装 Unsloth

```bash
# 在 Kaggle 中运行
!pip install unsloth
```

### 4. 运行官方 Notebook

```bash
# 直接打开 Colab 链接即可运行
# 例如：Llama 3.1 (8B) Alpaca 示例
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-Alpaca.ipynb
```

> 📁 **完整代码**：`unsloth-finetuning-guide/verify_installation.py`

---

## 第二阶段：数据集准备 - **重点！**

> **核心原则**：70% 的时间在洗数据，不是写代码。
> **100条精品数据原则**：先用100条高质量数据验证流程，再扩展到1000条。

### 1. 数据集格式要求

#### 格式1：Alpaca 格式（推荐）

```json
{
    "instruction": "解释LoRA的工作原理",
    "input": "",
    "output": "LoRA通过在预训练模型的权重矩阵旁边添加两个小矩阵A和B，只训练这两个小矩阵，从而大幅减少可训练参数数量..."
}
```

#### 格式2：ShareGPT 格式（多轮对话）

```json
{
    "conversations": [
        {"from": "human", "value": "什么是QLoRA？"},
        {"from": "gpt", "value": "QLoRA是..."}
    ]
}
```

### 2. 数据集准备代码

```python
from datasets import Dataset

# 手动准备数据集
data = {
    "instruction": ["任务1", "任务2", "任务3"],
    "input": ["输入1", "输入2", "输入3"],
    "output": ["输出1", "输出2", "输出3"]
}
dataset = Dataset.from_dict(data)

# 划分训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
```

> 📁 **完整代码**：`unsloth-finetuning-guide/prepare_data.py`

### 3. 使用 Unsloth Data Recipes（推荐）

```python
# Unsloth 支持自动从 PDF/CSV/JSON 生成 QA 对
# 参考文档：https://unsloth.ai/docs/new/studio/data-recipe
```

### 4. 数据质量检查清单

- [ ] 去除重复数据
- [ ] 检查回答一致性
- [ ] 验证数据格式
- [ ] 划分训练集/验证集（80/20）
- [ ] 检查是否有前后矛盾的问答对

---

## 第三阶段：Qwen2.5 微调实战

### 1. 基础微调代码

```python
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 加载模型（QLoRA）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # 自动检测
    load_in_4bit=True,
)

# 2. 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth 优化
)

# 3. 格式化数据集
def format_prompt(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

# 应用格式化
train_dataset = train_dataset.map(lambda x: {"text": format_prompt(x)})
eval_dataset = eval_dataset.map(lambda x: {"text": format_prompt(x)})

# 4. 训练配置
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./qwen2.5-finetuned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        optim="adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=100,
    ),
)

# 5. 开始训练
trainer.train()

# 6. 保存模型
trainer.save_model("./qwen2.5-finetuned-final")
```

> 📁 **完整代码**：`unsloth-finetuning-guide/finetune_basic.py`

### 2. 训练监控要点

- **training_loss**：应该逐渐下降
- **验证损失**：如果上升，可能过拟合
- **GPU 利用率**：确保在 80% 以上
- **学习率**：观察是否按预期变化

### 3. 实验记录模板

| 实验编号 | 数据集 | rank | alpha | 学习率 | batch_size | epochs | 最终Loss | 备注 |
|----------|--------|------|-------|--------|------------|--------|----------|------|
| 1 | 数据集A | 16 | 32 | 2e-4 | 2 | 3 | | |
| 2 | 数据集B | 32 | 64 | 1e-4 | 4 | 2 | | |
| 3 | 数据集C | 8 | 16 | 5e-4 | 1 | 5 | | |

---

## 第四阶段：训练参数调优

### 1. 关键参数说明

```python
training_args = TrainingArguments(
    # 批次大小
    per_device_train_batch_size=2,      # 根据 VRAM 调整
    gradient_accumulation_steps=4,      # 模拟更大批次（有效批次=2*4=8）
    
    # 学习率
    learning_rate=2e-4,                 # LoRA 标准起点
    lr_scheduler_type="cosine",         # 余弦退火
    warmup_ratio=0.03,                  # 3% 预热
    
    # 训练轮次
    num_train_epochs=3,                 # 避免过拟合（1-3轮推荐）
    
    # 精度
    fp16=False,                         # 根据 GPU 支持
    bf16=True,                          # 推荐
    
    # 其他
    gradient_checkpointing=True,        # 节省 VRAM
    optim="adamw_8bit",                 # 8-bit 优化器
)
```

> 📁 **完整代码**：`unsloth-finetuning-guide/training_args_example.py`（含 8GB 显存优化配置）

### 2. 调优策略

#### 问题1：过拟合（训练损失下降，验证损失上升）
- 增加 `lora_dropout`（0.05→0.1）
- 减少 `num_train_epochs`
- 增加数据集规模
- 降低 `r`（如 16→8）

#### 问题2：欠拟合（损失不下降）
- 增加 `r`（如 16→32）
- 增加 `num_train_epochs`
- 提高 `learning_rate`（如 2e-4→1e-3）
- 检查数据质量

#### 问题3：VRAM 不足
- 减少 `per_device_train_batch_size`（2→1）
- 增加 `gradient_accumulation_steps`（4→8）
- 确保使用 QLoRA（`load_in_4bit=True`）
- 启用 `gradient_checkpointing=True`

#### 问题4：训练速度慢
- 启用 `use_gradient_checkpointing="unsloth"`
- 使用 Flash Attention 2
- 检查 GPU 利用率（`nvidia-smi`）

### 3. 云端 GPU 推荐配置

| 云端平台 | GPU VRAM | 模型大小 | 推荐配置 |
|----------|----------|----------|----------|
| Colab T4 | ≈15GB | 3B QLoRA | batch_size=4, grad_accum=2, r=16 |
| Colab T4 | ≈15GB | 7B QLoRA | batch_size=2, grad_accum=4, r=16 |
| Colab T4 | ≈15GB | 8B QLoRA | batch_size=2, grad_accum=4, r=16 |
| Kaggle 2×T4 | ≈30GB | 14B QLoRA | batch_size=2, grad_accum=4, r=16 |
| Kaggle 2×T4 | ≈30GB | 7B LoRA(16-bit) | batch_size=1, grad_accum=8, r=16 |
| 付费云 GPU | A100(40GB) | 13B+ LoRA | batch_size=4, grad_accum=2, r=32 |

---

## 第五阶段：模型导出与部署

### 1. 保存 LoRA adapter

```python
# 保存 adapter（小文件，通常 10-100MB）
model.save_pretrained("./qwen2.5-lora-adapter")

# 推送到 HuggingFace
model.push_to_hub("your-username/qwen2.5-finetuned")

# 保存 tokenizer
tokenizer.save_pretrained("./qwen2.5-lora-adapter")
```

> 📁 **完整代码**：`unsloth-finetuning-guide/save_model.py`

### 2. 导出为 GGUF（用于 Ollama/llama.cpp）

```python
# 保存为 GGUF 格式
model.save_pretrained_gguf(
    "./qwen2.5-gguf",
    tokenizer,
    quantization_method="q4_k_m"  # 推荐量化方法
)

# 其他量化选项
# quantization_method="q8_0"   # 8-bit，质量更高
# quantization_method="q5_k_m" # 5-bit，平衡选择
```

> 📁 **完整代码**：`unsloth-finetuning-guide/export_gguf.py`

### 3. 使用 Ollama 部署

```bash
# 创建 Modelfile
cat > Modelfile << 'EOF'
FROM ./qwen2.5-gguf/unsloth.Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

SYSTEM "你是一个专业的助手，能够准确回答各种问题。"
EOF

# 创建 Ollama 模型
ollama create qwen2.5-finetuned -f Modelfile

# 运行模型
ollama run qwen2.5-finetuned
```

### 4. 使用 Unsloth Studio 部署

```bash
# 启动 Unsloth Studio
unsloth studio

# 通过 Web UI 运行和测试模型
# 支持 API 端点：http://localhost:8000/v1
```

---

## 实验记录与对比

### 实验目标

完成以下 3 次微调实验，记录关键指标：

#### 实验1：基线实验
- 数据集：小规模高质量数据（100条）
- 配置：r=16, alpha=32, lr=2e-4, epochs=3
- 目标：验证流程，建立基线

#### 实验2：数据规模实验
- 数据集：中等规模数据（500条）
- 配置：r=16, alpha=32, lr=2e-4, epochs=3
- 目标：观察数据量对效果的影响

#### 实验3：参数调优实验
- 数据集：同实验2
- 配置：r=32, alpha=64, lr=1e-4, epochs=2
- 目标：验证参数调整的效果

### 评估方法

1. **定量评估**：
   - 训练损失曲线
   - 验证损失
   - 困惑度（Perplexity）

2. **定性评估**：
   - 手动测试 10-20 个代表性问题
   - 检查回答的准确性、相关性、流畅性
   - 与基础模型对比

---

## 常见问题与解决方案

### Q1: VRAM 不足（CUDA out of memory）
```python
# 解决方案
model = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",  # 使用 4-bit 量化
    load_in_4bit=True,
)
# 或减少 batch_size，增加 gradient_accumulation_steps
```

### Q2: 训练速度慢
```python
# 解决方案
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing="unsloth",  # 启用 Unsloth 优化
)
# 检查是否安装了正确的 Triton 版本
```

### Q3: 模型输出质量差
- 检查数据质量（最重要！）
- 调整 LoRA rank（16→32→64）
- 调整学习率（2e-4→1e-4→5e-5）
- 增加训练轮次

### Q4: 如何选择模型？
- **入门推荐**：`unsloth/Qwen2.5-7B-bnb-4bit`
- **高质量需求**：`unsloth/Qwen2.5-14B-bnb-4bit`
- **中文优化**：`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`

---

## Unsloth Free Notebooks 完整列表（250+）

> 所有 Notebook 均可在 Google Colab 免费运行，推荐 T4 GPU（15GB VRAM）起步
>
> 完整列表：https://github.com/unslothai/notebooks | 文档：https://unsloth.ai/docs/get-started/unsloth-notebooks

### 基础微调 Notebooks

| 模型 | 类型 | Colab 链接 | 说明 |
|------|------|-----------|------|
| **Llama 3.1 (8B)** | Alpaca 微调 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-Alpaca.ipynb) | 入门首选，Alpaca 数据格式 |
| **Llama 3.2 (1B + 3B)** | 对话微调 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_%281B_and_3B%29-Conversational.ipynb) | 小模型，适合低显存 |
| **Qwen3 (14B)** | 推理对话 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_%2814B%29-Reasoning-Conversational.ipynb) | 中文优化，推理能力强 |
| **Phi-4 (14B)** | 对话微调 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb) | 微软模型，性价比高 |
| **Mistral v0.3 (7B)** | Alpaca 微调 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_%287B%29-Alpaca.ipynb) | Mistral 系列经典 |
| **gpt-oss (20B)** | 微调 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-%2820B%29-Fine-tuning.ipynb) | 大模型微调示例 |

### GRPO & 强化学习 Notebooks

| 模型 | 任务 | Colab 链接 | 说明 |
|------|------|-----------|------|
| **Qwen3 (4B)** | DAPO Math | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_%284B%29-GRPO.ipynb) | GRPO 强化学习入门 |
| **Qwen3 (8B)** | DAPO Math + vLLM | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_8B_FP8_GRPO.ipynb) | FP8 精度训练 |
| **Llama3.1 (8B)** | GSM8K Math | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-GRPO.ipynb) | 数学推理强化 |
| **gpt-oss (20B)** | GRPO | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-%2820B%29-GRPO.ipynb) | 大模型强化学习 |
| **Llama3 (8B)** | ORPO | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_%288B%29-ORPO.ipynb) | 偏好优化训练 |
| **Zephyr (7B)** | DPO | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_%287B%29-DPO.ipynb) | 直接偏好优化 |
| **Muse Glimmer (30B)** | Sudoku GRPO | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Muse_Glimmer_%2830B%29-GRPO.ipynb) | 游戏强化学习 |
| **Gemma4 (E2B)** | Sudoku/2048 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_%28E2B%29_GRPO.ipynb) | 多种游戏 RL |

### 视觉/多模态 Notebooks

| 模型 | 任务 | Colab 链接 | 说明 |
|------|------|-----------|------|
| **Gemma 4 (E2B)** | Vision | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_%28E2B%29-Vision.ipynb) | 视觉理解 |
| **Gemma 4 (31B)** | Vision | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_%2831B%29-Vision.ipynb) | 大参数视觉模型 |
| **Qwen3.5 (4B)** | Vision | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_%284B%29_Vision.ipynb) | 中文视觉理解 |
| **Qwen3-VL (8B)** | Vision | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_%288B%29-Vision.ipynb) | 视觉语言模型 |
| **Qwen2.5 VL (7B)** | Vision Math GRPO | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_5_7B_VL_GRPO.ipynb) | 视觉数学推理 |
| **Llama3.2 (11B)** | Vision | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_%2811B%29-Vision.ipynb) | Llama 视觉模型 |
| **Gemma3 (4B)** | Vision | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_%284B%29-Vision.ipynb) | 轻量视觉模型 |

### TTS 语音合成 Notebooks

| 模型 | Colab 链接 | 说明 |
|------|-----------|------|
| **Orpheus (3B)** | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_%283B%29-TTS.ipynb) | 高质量 TTS |
| **Spark TTS (0.5B)** | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Spark_TTS_%280_5B%29.ipynb) | 轻量 TTS |
| **Llasa TTS (1B)** | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llasa_TTS_%281B%29.ipynb) | TTS 微调 |
| **Llasa TTS (3B)** | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llasa_TTS_%283B%29.ipynb) | 大参数 TTS |
| **Sesame CSM (1B)** | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Sesame_CSM_%281B%29-TTS.ipynb) | 对话式 TTS |

### Embedding 嵌入模型 Notebooks

| 模型 | 任务 | Colab 链接 | 说明 |
|------|------|-----------|------|
| **embeddinggemma (300M)** | Embeddings | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma_%28300M%29.ipynb) | 轻量嵌入模型 |
| **Qwen3-Embedding (0.6B)** | Embeddings | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_Embedding_%280_6B%29.ipynb) | 中文嵌入 |
| **Qwen3-Embedding (4B)** | Embeddings | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_Embedding_%284B%29.ipynb) | 大参数嵌入 |
| **ModernBert** | Classification | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/ModernBert.ipynb) | 文本分类 |

### 工具调用 Notebooks

| 模型 | 任务 | Colab 链接 | 说明 |
|------|------|-----------|------|
| **Qwen2.5 Coder (1.5B)** | Tool Calling | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_%281.5B%29-Tool_Calling.ipynb) | 代码工具调用 |
| **FunctionGemma (270M)** | Tool Calling | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_%28270M%29-Multi-Turn-Tool-Calling.ipynb) | 多轮工具调用 |

### Unsloth Studio

| 类型 | Colab 链接 | 说明 |
|------|-----------|------|
| **Unsloth Studio** | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Unsloth_Studio.ipynb) | Web UI 微调界面 |

---

### 推荐学习路径（按 Notebook）

| 阶段 | 推荐 Notebook | 目标 |
|------|--------------|------|
| **入门** | Llama 3.1 (8B) Alpaca | 理解基本微调流程 |
| **进阶** | Qwen3 (4B) GRPO | 学习强化学习 |
| **视觉** | Qwen3.5 (4B) Vision | 掌握多模态微调 |
| **TTS** | Orpheus (3B) TTS | 语音合成微调 |
| **部署** | Unsloth Studio | 无代码微调体验 |

---

## 学习资源

### 官方资源
- Unsloth 文档：https://unsloth.ai/docs
- 微调指南：https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- Notebook 示例：https://github.com/unslothai/notebooks
- 模型目录：https://unsloth.ai/docs/get-started/unsloth-model-catalog

### 社区支持
- Discord：https://discord.gg/unsloth
- Reddit：https://reddit.com/r/unsloth
- GitHub Issues：https://github.com/unslothai/unsloth/issues

### 相关工具
- Ollama：https://ollama.com（本地计算机部署）
- LM Studio：https://lmstudio.ai（GUI 界面）
- HuggingFace：https://huggingface.co（模型库）

---

## 阶段结束检查清单

- [ ] 完成 Unsloth 安装和环境配置
- [ ] 运行至少 2 个官方 Free Notebook（Llama 3.1 Alpaca + Qwen3 GRPO）
- [ ] 准备 3 个不同质量/规模的数据集
- [ ] 完成 3 次 Qwen2.5 微调实验
- [ ] 记录每次实验的参数和结果
- [ ] 导出模型为 GGUF 格式
- [ ] 使用 Ollama 在本地计算机部署模型
- [ ] 对比 Unsloth 与原生 HuggingFace 的速度差异
- [ ] 为下一步 LlamaFactory 学习做好准备

---

## 下一步学习

完成 Unsloth 学习后，将进入 LlamaFactory 学习：
- 使用 LlamaFactory 复现 Unsloth 的结果
- 对比两者速度差异
- 理解工业化框架如何封装底层细节
- 学习蒸馏技术

---

> **关键提醒**：弃坑高峰期在洗数据阶段（洗到吐），熬过去你就超过 80% 只会跑 Demo 的人。
> 
> **试错比看书重要**：显卡能点亮就直接跑 Demo，遇红字报错再搜方案，比啃完《深度学习》再碰代码快10倍。