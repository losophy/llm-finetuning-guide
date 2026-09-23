# Unsloth 微调实战指南

> **运行方式**：训练与导出全部在云端 GPU 完成（Google Colab 或 Kaggle 免费额度），本地不装 CUDA、不下模型权重、不占磁盘；只在最后一步用**本地已装的 Ollama** 运行导出的模型。
>
> **学习目标**：掌握 Unsloth 微调框架，能独立完成模型的微调、参数调优、GGUF 导出与本地部署
>
> **前置知识**：已掌握 LoRA/QLoRA 基础理论（参见 `LoRA与QLoRA微调大语言模型完整指南.md`）

---

## 怎么用这份指南

**先跑通，再理解。** Unsloth 官方明确建议初学者从预置 Notebook 起步——先把整条链路完整跑一遍，再回头抠细节，比先啃概念快得多。

| 步骤 | 做什么 | 在哪做 | 产出 |
|------|--------|--------|------|
| **第 0 步** | 跑一个官方 Notebook，**一行代码都不改** | Colab / Kaggle | 建立"完整流程长什么样"的全局印象 |
| 第 1 步 | 云端环境准备（挂载存储、装 Unsloth） | 云端 | 可用的训练环境 |
| 第 2 步 | 准备数据集（Alpaca / ShareGPT） | 云端 | 训练集 + 验证集 |
| 第 3 步 | 用 Qwen2.5-7B 跑通一次 QLoRA 微调 | 云端 | LoRA adapter |
| 第 4 步 | 参数调优、记录实验 | 云端 | 可对比的实验结果 |
| 第 5 步 | 导出 GGUF → 下载到本地 → Ollama 运行 | 云端导出 / 本地运行 | 本地可对话的模型 |

### 第 0 步：先跑一个官方 Notebook

打开下面任一链接（点开就是可运行 Notebook），**不改任何代码**，按顺序把单元格跑完：

- **Llama 3.1 (8B) Alpaca**（入门首选）
  https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-Alpaca.ipynb
- **Llama 3.2 (1B + 3B) 对话**（模型更小，跑得更快）
  https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_%281B_and_3B%29-Conversational.ipynb

跑完后你会看到：**加载模型 → 格式化数据 → 训练 → 推理测试**，这四件事在一份 Notebook 里全干完了。这就是后面 5 个步骤要展开的内容。

> **平台说明**：Colab 点开即可运行；Kaggle 需先**验证账号**才能同时开启 GPU 和 Internet。

---

## 一、云端平台与显存门槛

### 1.1 Colab 与 Kaggle 怎么选

| | Google Colab | Kaggle |
|---|---|---|
| GPU | 单 T4（≈15GB） | 2×T4（≈30GB，默认只用单卡） |
| 免费额度 | 每天限时、不保证 | 每周 30 小时，更可预期 |
| 持久化 | 挂载 Google Drive | 写入 `/kaggle/working`（会话结束自动保存为 Output）或存成 Dataset |
| 联网 | 默认可用 | **需先验证账号**才能同时开 GPU 和 Internet |
| 适合 | 快速起步、跟官方 Notebook | 时间较长的训练（7B 多轮实验） |

**推荐路径**：入门用 Colab（点开官方 Notebook 即跑）→ 需要长时间训练时换 Kaggle。

> 两个平台**代码零改动**：本文示例在 Colab 和 Kaggle 上都能直接跑，只有"数据放哪、怎么续训"的写法不同（见 2.3）。

### 1.2 显存门槛（QLoRA 4-bit）

| 模型参数 | 最低显存 | T4（≈15GB）能跑 | 说明 |
|----------|----------|------------------|------|
| 3B | 3.5 GB | ✅ 轻松 | 入门首选 |
| **7B** | **5 GB** | ✅ 轻松 | **本文主线模型** |
| 8B | 6 GB | ✅ 轻松 | Llama 3.1 等 |
| 9B | 6.5 GB | ✅ 轻松 | — |
| 14B | 8.5 GB | ✅ 可跑 | Phi-4、Qwen3 14B 等 |
| 20B | 10 GB+ | ⚠️ 勉强（QLoRA） | gpt-oss 20B 等 |
| 27B | 22 GB | ❌ | 需更大显存 |

### 1.3 推荐配置

| 模型 | 平台 | batch_size | grad_accum | max_seq_len | 说明 |
|------|------|------------|------------|-------------|------|
| 3B QLoRA | Colab T4 | 4 | 2 | 2048 | 从容运行 |
| **7B QLoRA** | **Colab T4** | **2** | **4** | **2048** | **本文默认配置** |
| 8B QLoRA | Colab T4 | 2 | 4 | 2048 | Llama 3.1 等 |
| 14B QLoRA | Kaggle 2×T4 | 2 | 4 | 2048 | 需更大显存 |
| 7B LoRA(16-bit) | Kaggle 2×T4 | 1 | 8 | 2048 | 19GB 显存需求 |

---

## 二、云端环境准备

### 2.1 Colab 环境配置

**步骤 1：选择 GPU 运行时**（菜单：代码执行程序 → 更改运行时类型 → T4 GPU）

```python
# 在 Colab 中运行，确认 GPU 可用
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**步骤 2：安装 Unsloth**

```bash
!pip install unsloth
!pip install --upgrade --no-cache-dir torch trl peft accelerate
```

**步骤 3：挂载 Google Drive**（用于持久化数据和检查点）

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/unsloth-project
```

### 2.2 Kaggle 环境配置

**步骤 1：开启 GPU 与 Internet**

在 Notebook 右侧设置面板中：Accelerator 选 **GPU T4 ×2**，并把 **Internet** 打开。两者都需要账号已完成验证。

**步骤 2：安装 Unsloth**

```bash
!pip install unsloth
```

### 2.3 存储与断线续训

免费额度随时会断，训练成果必须落在持久化存储里：

| | Colab | Kaggle |
|---|---|---|
| 放哪 | `/content/drive/MyDrive/unsloth-project`（需先 `drive.mount()`） | `/kaggle/working/`（会话结束时自动保存为 Output） |
| 重连后 | 重新 mount 即可访问 | 把 Output 挂载为新 Notebook 的输入数据 |

**控制 checkpoint 体积**（两个平台都适用，免费空间都不大）：

- **只保留最新一个 checkpoint**：训练参数里加 `save_total_limit=1`
- **训练完只下载 adapter**（几十 MB），不要下载整个 checkpoint 目录
- Colab 免费 Drive 只有 15GB，7B 的 checkpoint（含优化器状态）单个可能上 GB，多个叠加很快会满

### 2.4 验证环境

运行 `verify_installation.py` 检查 PyTorch / CUDA / GPU / VRAM 是否正常：

> 📁 **完整代码**：`unsloth-finetuning-guide/verify_installation.py`

---

## 三、数据集准备

> **核心原则**：70% 的时间在洗数据，不是写代码。
> **100 条精品原则**：先用 100 条高质量数据验证流程，再扩展到 1000 条。

### 3.1 两种数据格式

**格式 1：Alpaca 格式（推荐）**

```json
{
    "instruction": "解释LoRA的工作原理",
    "input": "",
    "output": "LoRA通过在预训练模型的权重矩阵旁边添加两个小矩阵A和B，只训练这两个小矩阵，从而大幅减少可训练参数数量..."
}
```

**格式 2：ShareGPT 格式（多轮对话）**

```json
{
    "conversations": [
        {"from": "human", "value": "什么是QLoRA？"},
        {"from": "gpt", "value": "QLoRA是..."}
    ]
}
```

### 3.2 加载与划分

> **数据集来源**：本指南不附带数据文件。默认加载 HuggingFace Hub 的公开数据集 `yahma/alpaca-cleaned`（Alpaca 格式），可换成中文数据集（如 `shibing624/alpaca-zh`），也可传入本地 JSONL——脚本会自动识别 Alpaca / ShareGPT 两种字段。
> 中国大陆网络可先设镜像：`export HF_ENDPOINT=https://hf-mirror.com`

```python
from prepare_data import prepare_datasets

# 默认加载 HuggingFace 数据集；也可传本地文件：prepare_datasets("data/xxx.jsonl")
train_dataset, eval_dataset = prepare_datasets()
```

> 📁 **完整代码**：`unsloth-finetuning-guide/prepare_data.py`

### 3.3 使用 Unsloth Data Recipes（可选）

Unsloth 支持从 PDF/CSV/JSON 自动生成 QA 对，参考文档：https://unsloth.ai/docs/new/studio/data-recipe

### 3.4 数据质量检查清单

- [ ] 去除重复数据
- [ ] 检查回答一致性
- [ ] 验证数据格式
- [ ] 划分训练集/验证集（80/20）
- [ ] 检查是否有前后矛盾的问答对

---

## 四、微调实战

### 4.1 完整训练代码

```python
"""基础微调脚本 - Qwen2.5 QLoRA 微调示例"""

import torch
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer


def load_model(model_name="unsloth/Qwen2.5-7B-bnb-4bit", max_seq_length=2048):
    """加载模型（QLoRA 4-bit）"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # 自动检测
        load_in_4bit=True,
    )
    return model, tokenizer


def configure_lora(model):
    """配置 LoRA 参数"""
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
    return model


def format_prompt(sample):
    """格式化 Alpaca 格式数据（input 为空时省略 Input 段）"""
    if str(sample.get("input", "")).strip():
        return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""


def create_trainer(model, tokenizer, train_dataset, eval_dataset,
                   max_length=2048, output_dir="./qwen2.5-finetuned"):
    """创建训练器"""
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            max_length=max_length,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_steps=100,
            optim="adamw_8bit",
            warmup_steps=10,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=100,
        ),
    )
    return trainer


def train(trainer, output_dir="./qwen2.5-finetuned-final"):
    """开始训练并保存模型"""
    trainer.train()
    trainer.save_model(output_dir)
    return trainer


if __name__ == "__main__":
    from prepare_data import prepare_datasets

    # 加载模型
    model, tokenizer = load_model()

    # 配置 LoRA
    model = configure_lora(model)

    # 准备数据集（默认 HuggingFace 公开数据集；也可传本地 JSONL 路径）
    train_dataset, eval_dataset = prepare_datasets()
    train_dataset = train_dataset.map(lambda x: {"text": format_prompt(x)})
    eval_dataset = eval_dataset.map(lambda x: {"text": format_prompt(x)})

    # 创建训练器
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset)

    # 开始训练
    train(trainer)

    print("训练完成，模型已保存到 ./qwen2.5-finetuned-final")
```

> 📁 **完整代码**：`unsloth-finetuning-guide/finetune_basic.py`（本代码块与脚本逐字一致）

### 4.2 训练监控要点

- **training_loss**：应该逐渐下降
- **验证损失**：如果上升，可能过拟合
- **GPU 利用率**：确保在 80% 以上
- **学习率**：观察是否按预期变化

### 4.3 实验记录模板

| 实验编号 | 数据集 | rank | alpha | 学习率 | batch_size | epochs | 最终Loss | 备注 |
|----------|--------|------|-------|--------|------------|--------|----------|------|
| 1 | 100 条精品 | 16 | 32 | 2e-4 | 2 | 3 | | 基线 |
| 2 | 500 条 | 16 | 32 | 2e-4 | 2 | 3 | | 数据规模对比 |
| 3 | 500 条 | 32 | 64 | 1e-4 | 2 | 2 | | 参数调优对比 |

---

## 五、参数调优与排错

### 5.1 关键参数说明

```python
training_args = TrainingArguments(
    # 批次大小
    per_device_train_batch_size=2,      # 根据 VRAM 调整
    gradient_accumulation_steps=4,      # 模拟更大批次（有效批次=2*4=8）

    # 学习率
    learning_rate=2e-4,                 # LoRA 标准起点
    lr_scheduler_type="cosine",         # 余弦退火
    warmup_steps=10,                    # 预热步数（新版已移除 warmup_ratio）

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

### 5.2 调优策略

#### 问题 1：过拟合（训练损失下降，验证损失上升）

- 增加 `lora_dropout`（0.05→0.1）
- 减少 `num_train_epochs`
- 增加数据集规模
- 降低 `r`（如 16→8）

#### 问题 2：欠拟合（损失不下降）

- 增加 `r`（如 16→32）
- 增加 `num_train_epochs`
- 提高 `learning_rate`（如 2e-4→1e-3）
- 检查数据质量

#### 问题 3：VRAM 不足（CUDA out of memory）

- 减少 `per_device_train_batch_size`（2→1）
- 增加 `gradient_accumulation_steps`（4→8）
- 确保使用 QLoRA（`load_in_4bit=True`）
- 确认 `use_gradient_checkpointing="unsloth"` 已开启

#### 问题 4：训练速度慢

- 确认 `use_gradient_checkpointing="unsloth"`（这是 Unsloth 自带的优化内核，**不需要手动安装 Flash Attention 2**——官方不建议在 Windows 上手动编译 FA2）
- 检查 GPU 利用率（`nvidia-smi`）
- 缩短 `max_seq_length`、减小数据集规模

#### 问题 5：模型输出质量差

- 检查数据质量（最重要！）
- 调整 LoRA rank（16→32→64）
- 调整学习率（2e-4→1e-4→5e-5）
- 增加训练轮次

#### 问题 6：如何选择模型？

- **入门推荐**：`unsloth/Qwen2.5-7B-bnb-4bit`
- **高质量需求**：`unsloth/Qwen2.5-14B-bnb-4bit`
- **中文优化**：`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`

### 5.3 云端推荐配置

| 云端平台 | GPU VRAM | 模型大小 | 推荐配置 |
|----------|----------|----------|----------|
| Colab T4 | ≈15GB | 3B QLoRA | batch_size=4, grad_accum=2, r=16 |
| Colab T4 | ≈15GB | 7B QLoRA | batch_size=2, grad_accum=4, r=16 |
| Colab T4 | ≈15GB | 8B QLoRA | batch_size=2, grad_accum=4, r=16 |
| Kaggle 2×T4 | ≈30GB | 14B QLoRA | batch_size=2, grad_accum=4, r=16 |
| Kaggle 2×T4 | ≈30GB | 7B LoRA(16-bit) | batch_size=1, grad_accum=8, r=16 |

---

## 六、导出与部署

> **分工**：云端负责**导出**，本地负责**运行**。云端**不需要安装 Ollama**——GGUF 的生成由 Unsloth 完成，Ollama 只是本地用来加载它的工具。

### 6.1 保存 LoRA adapter（云端）

```python
# 保存 adapter（小文件，通常几十 MB）
model.save_pretrained("./qwen2.5-lora-adapter")
tokenizer.save_pretrained("./qwen2.5-lora-adapter")

# 可选：推送到 HuggingFace
# model.push_to_hub("your-username/qwen2.5-finetuned")
```

> 📁 **完整代码**：`unsloth-finetuning-guide/save_model.py`

### 6.2 导出为 GGUF（云端）

```python
# 保存为 GGUF 格式
model.save_pretrained_gguf(
    "./qwen2.5-gguf",
    tokenizer,
    quantization_method="q4_k_m"  # 推荐量化方法
)
```

产物：`./qwen2.5-gguf/unsloth.Q4_K_M.gguf`

**量化方式对照（7B 模型）**：

| 量化方式 | 体积 | 说明 |
|----------|------|------|
| `q4_k_m` | ≈4.4 GB | **推荐**，质量与体积平衡 |
| `q5_k_m` | ≈5.4 GB | 质量更高 |
| `q8_0` | ≈8 GB | 接近原精度 |
| `q3_k_m` / `q2_k` | ≈3.5 / 2.8 GB | 更省空间，质量有损 |

> ⚠️ 导出时 Unsloth 会先把模型合并到 fp16 再量化，Colab 免费版的系统内存比较紧，7B 可能内存不足；如遇报错，换 Kaggle（内存更宽裕）或改用 `q3_k_m`。

> 📁 **完整代码**：`unsloth-finetuning-guide/export_gguf.py`

### 6.3 下载到本地

把上一步生成的 `.gguf`（约 4.4GB）从云端下载到本地。建议先落 Google Drive / Kaggle Output 再下载，避免会话中断导致文件丢失。

### 6.4 本地使用 Ollama 运行

前提：本地已安装 Ollama。创建 `Modelfile`（`FROM` 路径按实际存放位置填写）：

```bash
cat > Modelfile << 'EOF'
FROM D:/models/unsloth.Q4_K_M.gguf

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

> 💡 **省空间要点**：`ollama create` 会把 GGUF **复制**进 Ollama 自己的模型库（Windows 下在 `C:\Users\<用户名>\.ollama\models`），不是引用原文件。
> 所以刚导入完时本地会有**两份**（合计约 9GB）。**导入成功后，原始的那个 `.gguf` 就可以删掉**，最终只占约 4.4GB。

### 6.5 云端 ↔ 本地分工一览

| 环节 | 在哪做 | 需要什么 |
|------|--------|----------|
| 环境准备 / 训练 / 调参 | 云端（Colab 或 Kaggle） | 免费 GPU 额度 |
| 导出 adapter / GGUF | 云端 | Unsloth（**不需要 Ollama**） |
| 下载 GGUF | 本地 | 约 4.4GB 磁盘空间 |
| 加载并对话 | 本地 | 已安装的 Ollama |

---

## 七、附录：官方 Notebook 清单（250+）

> 所有 Notebook 均可在 Google Colab 免费运行，按需取用；完整列表：https://github.com/unslothai/notebooks
> 文档：https://unsloth.ai/docs/get-started/unsloth-notebooks

### 基础微调 Notebooks

| 模型 | 类型 | Colab 链接 | 说明 |
|------|------|-----------|------|
| **Llama 3.1 (8B)** | Alpaca 微调 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-Alpaca.ipynb) | 入门首选，Alpaca 数据格式 |
| **Llama 3.2 (1B + 3B)** | 对话微调 | [Open](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_%281B_and_3B%29-Conversational.ipynb) | 小模型，跑得快 |
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

### 视觉 / 多模态 Notebooks

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

## 学习资源

### 官方资源
- Unsloth 文档：https://unsloth.ai/docs
- 微调入门指南：https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- Notebook 示例：https://github.com/unslothai/notebooks
- 模型目录：https://unsloth.ai/docs/get-started/unsloth-model-catalog

### 社区支持
- Discord：https://discord.gg/unsloth
- Reddit：https://reddit.com/r/unsloth
- GitHub Issues：https://github.com/unslothai/unsloth/issues

### 相关工具
- Ollama：https://ollama.com（本地部署，本文第 6.4 节使用）
- LM Studio：https://lmstudio.ai（GUI 界面）
- HuggingFace：https://huggingface.co（模型库）

---

## 完成检查清单

- [ ] 跑通至少 2 个官方 Notebook（Llama 3.1 Alpaca + Qwen3 GRPO）
- [ ] 云端环境可用（Colab 或 Kaggle，含 Unsloth 安装与存储挂载）
- [ ] 准备好数据集（Alpaca / ShareGPT，或自己的数据）
- [ ] 完成 Qwen2.5-7B 的 QLoRA 微调，产出 LoRA adapter
- [ ] 记录每次实验的参数与 loss
- [ ] 导出 GGUF（q4_k_m）
- [ ] 下载到本地，用 Ollama 跑起来
- [ ] 对比 Unsloth 与原生 HuggingFace 的速度 / 显存差异
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
> **试错比看书重要**：云端环境能跑起来就直接跑官方 Notebook，遇红字报错再搜方案，比啃完《深度学习》再碰代码快 10 倍。
