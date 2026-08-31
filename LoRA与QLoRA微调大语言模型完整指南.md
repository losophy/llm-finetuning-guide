# 使用 LoRA 和 QLoRA 在 Python 中微调大语言模型——完整指南

> 原文：[Fine-Tuning LLMs with LoRA and QLoRA in Python — A Complete Guide](https://machinelearningplus.com/deep-learning/fine-tuning-llms-lora-qlora-python/)
> 作者：Selva Prabhakaran | 阅读时长约 27 分钟

使用 LoRA 和 QLoRA 在 Python 中微调大语言模型。本完整指南涵盖内存计算、PEFT 配置、4-bit QLoRA、Adapter 合并以及常见错误——全部附带可运行的代码。

> 📂 **配套代码：** 本指南每个代码块都有对应的可运行脚本，存放在本仓库的 `llm-lora-qlora-finetuning-guide/` 目录下，代码块上方均标注了对应文件，复制对应脚本即可直接运行。注意：配套脚本针对本地 trl 1.12 / transformers 5.x 环境做了参数适配（如 `max_length`、`warmup_steps`、`loss_type="nll"`、`processing_class`），与文中按 trl 0.8 API 书写的代码略有差异，功能等价。

---

GPT-4 不懂你公司的内部词汇。Llama 3 无法回答关于你私有数据集的问题。完全重新训练需要数万 GPU 小时，成本远超大多数团队的预算。

LoRA 和 QLoRA 弥补了这一差距。在单张消费级 GPU 上用几个小时微调一个 7B 模型——并在你的特定任务上超越基础模型。我见过这个模式在数十个领域适配项目中稳定奏效：关键在于知道该用哪种技术，以及为什么。

---

**在写一行代码之前，先了解各部分是如何拼在一起的。**

你从一个大规模预训练模型开始——数十亿个冻结权重，它们已经理解语言。模型的知识不是问题所在。问题在于模型从未见过你的领域。

全参数微调（Full fine-tuning）会解冻每一个权重并全部重新训练。对于 7B 模型，这意味着权重、梯度和优化器状态大约需要 98 GB 的 GPU 内存。大多数团队负担不起这样的硬件。

LoRA 采用了不同的方法。它问的是：*教会模型你的任务所需的最小改变是什么？* 它在每个冻结的权重层旁边注入两个微小的矩阵。训练期间，只有这些微小的矩阵会更新。基础模型从不改变。

QLoRA 更进一步。它将冻结的基础模型压缩为 4-bit 精度——将其内存占用削减 75%。LoRA adapter 仍然以 16-bit 精度训练，和之前一样。结果：一个 7B 模型在不到 6 GB 的 GPU 内存下就能放下。

学完本指南，你将理解这两种技术背后的数学原理，从零配置它们，端到端地训练一个模型，并知道何时该选 LoRA 而不是 QLoRA。

---

## 前置条件

- **Python 版本：** 3.9+
- **所需库：** torch (2.0+)、transformers (4.40+)、peft (0.10+)、trl (0.8+)、bitsandbytes (0.43+)、datasets (2.18+)、accelerate (0.28+)
- **安装：**

```bash
pip install torch transformers peft trl bitsandbytes datasets accelerate
```

- **硬件：** LoRA——12+ GB 显存的 GPU。QLoRA——6+ GB 显存（免费的 Colab T4 即可）。
- **背景知识：** Python、基础 PyTorch 张量、熟悉 Transformer 模型。
- **完成时间：** 60 分钟

---

> 📄 **对应代码文件：** [check_gpu.py](llm-lora-qlora-finetuning-guide/check_gpu.py) —— 本教程的全部导入与 GPU 环境检查

```python
# 本教程的全部导入——先运行这个单元
import os
import torch
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

```python
PyTorch version: 2.2.0
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
GPU memory: 24.0 GB
```

---

## 为什么全参数微调不可行（以及 LoRA 如何解决）

这里有个好用的思考方式。想象一位世界级的全科医生，他精通医学的一切。你需要他专攻只在你们地区出现的罕见儿科疾病。有两个选项：

1. 让他再上一次完整的医学院——8 年、花费巨大、还有遗忘通科知识的风险。
2. 让他在你们的儿科病房进行 3 个月的高强度轮转——目标明确、成本低，而且保留了他已有的知识。

全参数微调是选项 1。LoRA 是选项 2。

内存问题非常具体。让我们为一个 70 亿参数的模型算一下——这是当下微调实验最常见的规模。

> 📄 **对应代码文件：** [lora_vs_full_finetune_memory_calculator.py](llm-lora-qlora-finetuning-guide/lora_vs_full_finetune_memory_calculator.py) —— 计算全参数微调 vs LoRA 所需 GPU 内存

```python
# 计算全参数微调 vs LoRA 所需的 GPU 内存
# 纯算术——运行此单元不需要 GPU

model_params = 7_000_000_000  # 70 亿参数

bytes_per_param_bf16 = 2  # 16-bit = 2 字节
bytes_per_param_fp32 = 4  # 32-bit = 4 字节

# 全参数微调：权重 + 梯度 + Adam 优化器状态
# Adam 会存储动量 + 方差 = 每个梯度的 2 份额外 fp32 副本
weights_gb = model_params * bytes_per_param_bf16 / 1e9
gradients_gb = model_params * bytes_per_param_fp32 / 1e9
optimizer_gb = model_params * bytes_per_param_fp32 * 2 / 1e9
total_full_ft_gb = weights_gb + gradients_gb + optimizer_gb

print("=== 全参数微调 7B 模型所需内存 ===")
print(f"  权重 (bf16):         {weights_gb:.1f} GB")
print(f"  梯度 (fp32):       {gradients_gb:.1f} GB")
print(f"  Adam 优化器状态:  {optimizer_gb:.1f} GB")
print(f"  总计:                  {total_full_ft_gb:.1f} GB")

# LoRA 微调：只有 adapter 矩阵更新
# 32 个 Transformer 层 × 4 个注意力矩阵 = 128 个 adapter 对
lora_rank = 16
num_adapter_pairs = 32 * 4
hidden_dim = 4096

lora_params = num_adapter_pairs * 2 * (hidden_dim * lora_rank)

lora_weights_gb = lora_params * bytes_per_param_bf16 / 1e9
lora_grads_gb = lora_params * bytes_per_param_fp32 / 1e9
lora_optim_gb = lora_params * bytes_per_param_fp32 * 2 / 1e9
base_model_gb = model_params * bytes_per_param_bf16 / 1e9
total_lora_gb = base_model_gb + lora_weights_gb + lora_grads_gb + lora_optim_gb

print(f"\n=== LoRA 微调所需内存 (rank={lora_rank}, 仅注意力层) ===")
print(f"  冻结的基础模型 (bf16):  {base_model_gb:.1f} GB")
print(f"  LoRA 可训练参数:     {lora_params:,}")
print(f"  LoRA 权重 + 梯度 + 优化器: {(lora_weights_gb + lora_grads_gb + lora_optim_gb):.2f} GB")
print(f"  总计:                     {total_lora_gb:.1f} GB")
print(f"  相比全参数微调的内存缩减:      {total_full_ft_gb / total_lora_gb:.1f}x")
```

```python
=== 全参数微调 7B 模型所需内存 ===
  权重 (bf16):         14.0 GB
  梯度 (fp32):       28.0 GB
  Adam 优化器状态:  56.0 GB
  总计:                  98.0 GB

=== LoRA 微调所需内存 (rank=16, 仅注意力层) ===
  冻结的基础模型 (bf16):  14.0 GB
  LoRA 可训练参数:     16,777,216
  LoRA 权重 + 梯度 + 优化器: 0.24 GB
  总计:                     14.2 GB
  相比全参数微调的内存缩减:      6.9x
```

这 6.9 倍的缩减仅来自 LoRA 本身——而且我们只针对注意力层。把 MLP 层也算进去，adapter 的开销仍然不到 0.5 GB。

借助 QLoRA 对基础模型的 4-bit 压缩，7B 模型的总内存可降到 6 GB 以下。

> **LoRA 并不会压缩基础模型——它是通过大幅减少可训练参数来降低训练内存。** 基础模型仍然占用其全部内存。真正压缩基础模型的是 QLoRA。它们解决的是内存问题的不同部分。

**快速自测：** 如果你对 7B 模型应用 rank=8 而非 rank=16 的 LoRA，基础模型的内存会改变吗？（答案：不会。无论 rank 是多少，冻结的基础模型都不变。只有微小的 adapter 矩阵大小会变化。）

## LoRA 的工作原理——用通俗语言讲数学

在 Transformer 中，最重要的权重矩阵是注意力投影——Query（Q）、Key（K）、Value（V）和 Output（O）。对于 7B 模型，每个都是 4096 × 4096，即 1670 万个数值。全参数微调会调整所有这些数值。

LoRA 的洞见：你不需要改变全部 1670 万个数值。你只需要改变更新的一个*低秩近似*。

当微调将权重矩阵 W 移动一个变化量 ΔW 时，LoRA 将该变化近似为：

ΔW = B · A

其中：
- W 是原始冻结权重矩阵，形状为 (d_out × d_in)——例如 (4096 × 4096)
- A 是新的可训练矩阵，形状为 (r × d_in)——例如 (16 × 4096)
- B 是新的可训练矩阵，形状为 (d_out × r)——例如 (4096 × 16)
- r 是**秩（rank）**——一个较小的数字，如 4、8、16 或 32

与其训练 1670 万个参数，你只需训练 r × d_in + d_out × r = 131,072 个参数。单就这一层就实现了 128 倍的缩减。

*不擅长数学？直接跳过——下面的实际配置才是你需要的。*

下面的模拟具体展示了这一点。注意参数数量的缩减，并理解为什么 B 必须从零开始。

> 📄 **对应代码文件：** [lora_svd_simulation.py](llm-lora-qlora-finetuning-guide/lora_svd_simulation.py) —— 用 numpy 模拟 LoRA 的低秩分解

```python
# 用 numpy 模拟 LoRA 的低秩分解
np.random.seed(42)

d_out, d_in = 512, 512

# 全参数微调会产生的权重更新
delta_W_full = np.random.randn(d_out, d_in).astype(np.float32) * 0.01

# LoRA：用两个小矩阵 A 和 B 近似 delta_W
# B 初始化为 ZERO——模型一开始与预训练基础模型完全相同
rank = 16
A = np.random.randn(rank, d_in).astype(np.float32) * 0.01
B = np.zeros((d_out, rank), dtype=np.float32)

# 用 SVD 模拟训练后的 adapter——只保留前 'rank' 个奇异向量
# U 和 Vt 捕获最重要的方向；S 包含它们的幅度
U, S, Vt = np.linalg.svd(delta_W_full, full_matrices=False)
A_trained = np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]
B_trained = U[:, :rank] @ np.diag(np.sqrt(S[:rank]))
delta_W_lora = B_trained @ A_trained

print(f"原始权重矩阵: {d_out} × {d_in} = {d_out * d_in:,} 个数值")
print(f"\nLoRA 分解 (rank={rank}):")
print(f"  矩阵 A: {A_trained.shape}  →  {A_trained.size:,} 个参数")
print(f"  矩阵 B: {B_trained.shape}  →  {B_trained.size:,} 个参数")
print(f"  LoRA 总参数: {A_trained.size + B_trained.size:,}")
print(f"  全量更新:    {delta_W_full.size:,}")
print(f"  参数缩减: {delta_W_full.size / (A_trained.size + B_trained.size):.0f}x")
```

这部分一开始确实令人惊讶：重建误差看起来很高。LoRA 并不能完美复现完整的更新——它近似的是变化最重要的方向。原始的 LoRA 论文 [1] 表明，在大多数语言任务上，rank 4–16 可以达到或超过全参数微调的质量，尽管存在这种近似。你真正在乎的是参数数量：**对于一个 512×512 的层，只需训练 16,384 个数值，而不是 262,144 个。**

> **B 始终初始化为零。** 训练开始时，ΔW = B · A = 0。模型行为与预训练基础模型完全一致。训练从稳定、已知的起点开始。矩阵 A 使用小的随机值初始化，这样梯度从第一步就能流动。

## LoRA 超参数——每个参数的作用

四个参数控制你的 LoRA 配置。把它们设对比大多数人意识到的更重要。

**`r`（秩）** ——adapter 矩阵的大小。更高的秩 = 更多参数 = 更强的适应表达能力。常见取值：4、8、16、32。我几乎总是从 `r=16` 开始——它是 90% 任务的正确默认值，而且很少有必要超过 32。

**`lora_alpha`** ——前向传播时应用的缩放因子。模型计算 W + (α / r) · B · A。比值 α / r 控制 adapter 对输出的影响程度。**以 alpha = 2 × r 作为起点。** 如果 r=16，就设 alpha=32。这条规则来自 Lightning AI 的数百次实验 [3]，也与原始论文的建议一致 [2]。

**`lora_dropout`** ——LoRA 层上的 dropout 概率。防止在小数据集上过拟合。数据集小于 10,000 个样本时用 0.05–0.1。大数据集（50,000+）设为 0。

**`target_modules`** ——哪些权重矩阵挂上 LoRA adapter。针对所有线性层（Q、K、V、O，加上 MLP 门控）一贯优于只针对 Q 和 V [4]。额外的内存开销微乎其微，质量提升却是实实在在的。在看到全层覆盖带来的持续改进后，我不再只针对 Q+V。

听起来很熟悉？大多数教程仍然只推荐 Q+V。那是原始论文的做法。此后的实验证据明确指向全层覆盖。

> 📄 **对应代码文件：** [lora_parameter_counter.py](llm-lora-qlora-finetuning-guide/lora_parameter_counter.py) —— LoRA 秩与目标模块选择对可训练参数数量的影响

```python
# LoRA 秩和目标选择如何影响可训练参数数量
hidden_dim = 4096
intermediate_dim = 11008
num_layers = 32

def count_lora_params(rank, target="all"):
    attn_matrices = 4
    attn_params_per_layer = attn_matrices * 2 * (hidden_dim * rank)
    mlp_params_per_layer = (
        2 * (hidden_dim * rank + rank * intermediate_dim) +
        2 * (intermediate_dim * rank + rank * hidden_dim)
    )
    if target == "qv_only":
        params_per_layer = 2 * 2 * (hidden_dim * rank)
    elif target == "attention":
        params_per_layer = attn_params_per_layer
    else:
        params_per_layer = attn_params_per_layer + mlp_params_per_layer
    return params_per_layer * num_layers

print(f"{'秩':>6} | {'仅 QV':>12} | {'全部注意力':>12} | {'全部线性层':>12}")
print("-" * 55)
for rank in [4, 8, 16, 32, 64]:
    qv = count_lora_params(rank, "qv_only")
    attn = count_lora_params(rank, "attention")
    all_lin = count_lora_params(rank, "all")
    print(f"{rank:>6} | {qv/1e6:>10.2f}M | {attn/1e6:>10.2f}M | {all_lin/1e6:>10.2f}M")

total_rank16_all = count_lora_params(16, 'all')
print(f"\n7B 模型总参数: ~7,000M")
print(f"LoRA rank=16, 全部线性层: {total_rank16_all/1e6:.1f}M 可训练")
print(f"占总参数的 {total_rank16_all / 7e9 * 100:.2f}%")
```

```python
  秩 |      仅 QV |    全部注意力 |   全部线性层
-------------------------------------------------------
     4 |       2.10M |       4.19M |      11.93M
     8 |       4.19M |       8.39M |      23.86M
    16 |       8.39M |      16.78M |      47.71M
    32 |      16.78M |      33.55M |      95.42M
    64 |      33.55M |      67.11M |     190.84M

7B 模型总参数: ~7,000M
LoRA rank=16, 全部线性层: 47.7M 可训练
占总参数的 0.68%
```

---

## [动手练习] 练习 1：计算 13B 模型的 LoRA 参数

Llama 2 13B 模型的参数为：`hidden_dim = 5120`、`intermediate_dim = 13824`、`num_layers = 40`。

**你的任务：** 为 13B 模型修改 `count_lora_params` 函数。使用 `rank=16` 并针对所有线性层，计算可训练参数。再计算占 130 亿总参数的百分比。

*提示 1：复制该函数并更新顶部的三个维度常量。*
*提示 2：预期结果低于总参数的 1%。*

> 📄 **参考实现：** [count_lora_params_llama13b.py](llm-lora-qlora-finetuning-guide/count_lora_params_llama13b.py) —— 练习 1 的参考答案（13B 模型 LoRA 参数统计）

```python
# 练习 1：统计 Llama 2 13B 的 LoRA 参数
hidden_dim_13b = ___
intermediate_dim_13b = ___
num_layers_13b = ___
rank = 16

def count_lora_params_13b(rank):
    attn_params = 4 * 2 * (hidden_dim_13b * rank)
    mlp_params = (
        2 * (hidden_dim_13b * rank + rank * intermediate_dim_13b) +
        2 * (intermediate_dim_13b * rank + rank * hidden_dim_13b)
    )
    return (attn_params + mlp_params) * num_layers_13b

total_lora = count_lora_params_13b(rank)
total_model = 13_000_000_000
print(f"LoRA 可训练参数: {total_lora / 1e6:.1f}M")
print(f"LoRA 占比: {total_lora / total_model * 100:.2f}%")

# 答案：hidden_dim_13b=5120, intermediate_dim_13b=13824, num_layers_13b=40
# 预期：约 78.6M 可训练参数，占 13B 的 0.60%
```

---

## 使用 LoRA 微调——分步详解

理论有用。可运行的代码更好。让我们微调一个真实模型。

我们将使用 `facebook/opt-125m`——一个 1.25 亿参数的小模型——在指令跟随数据集上训练。OPT-125M 不是最先进的，但它足够小，可以在免费的 Colab GPU 上运行。你在这里写的代码与 Llama 3.1 8B 的代码**完全相同**。你只需换一个字符串，其他什么都不用改。

流程：加载模型 → 格式化数据集 → 配置 LoRA → 训练 → 保存 adapter。

### 第 1 步——加载基础模型和分词器

我们用 `bfloat16` 精度加载——内存只有 `float32` 的一半，数值稳定性还优于 `float16`。`device_map="auto"` 参数会自动把模型放到 GPU 上，必要时还能跨多张 GPU 拆分。

> 📄 **对应代码文件：** [opt-125m-lora-sandbox.py](llm-lora-qlora-finetuning-guide/opt-125m-lora-sandbox.py) —— 加载基础模型和分词器（流程验证用）

```python
model_name = "facebook/opt-125m"  # 生产环境换成 "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # OPT 没有单独的 pad token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型: {model_name}")
print(f"总参数: {total_params / 1e6:.1f}M")
print(f"可训练参数 (LoRA 之前): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
print(f"模型精度: {next(model.parameters()).dtype}")
```

```python
模型: facebook/opt-125m
总参数: 125.2M
可训练参数 (LoRA 之前): 125.2M
模型精度: torch.bfloat16
```

### 第 2 步——准备数据集

对于指令微调，每条训练样本需要一对"提示词-回答"，并格式化为单个文本字符串。SFTTrainer 会自动在损失中屏蔽提示词 token，让模型学习*生成回答*，而不是复制提示词。

自有数据最简单的格式：一个带 `text` 列的 CSV，每行包含完整的对话。下面的代码展示了如何把原始的问答列表转换成这种格式，然后加载一个预格式化好的示例数据集。

> 📄 **对应代码文件：** [prepare_sft_dataset.py](llm-lora-qlora-finetuning-guide/prepare_sft_dataset.py) —— 数据格式化与数据集加载

```python
# 把你自己的原始数据转换成 SFTTrainer 格式
# 每个 text 字符串把提示词 + 回答合并为一个字段
raw_qa_pairs = [
    {"question": "What is LoRA?", "answer": "LoRA is a parameter-efficient fine-tuning method..."},
    {"question": "How does QLoRA work?", "answer": "QLoRA combines 4-bit quantization with LoRA..."},
]

def format_instruction(sample):
    return f"### Human: {sample['question']}\n### Assistant: {sample['answer']}"

# 展示格式化后的文本长什么样
print("自定义数据集格式:")
print(format_instruction(raw_qa_pairs[0])[:200])

# 本教程使用一个预格式化好的公开数据集
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
print(f"\n数据集大小: {len(dataset)} 条样本")
print(f"\n预格式化示例 (前 250 个字符):")
print(dataset[0]["text"][:250])
```

```python
自定义数据集格式:
### Human: What is LoRA?
### Assistant: LoRA is a parameter-efficient fine-tuning method...

数据集大小: 9846

预格式化示例 (前 250 个字符):
### Human: Can you write a short introduction about the relevance of the term "monopsony" in economics?
### Assistant: "Monopsony" refers to a market structure where there is only one buyer...
```

### 第 3 步——用 PEFT 配置 LoRA

这是整个配置的核心。`LoraConfig` 告诉 PEFT 如何精确地注入 adapter 矩阵。`get_peft_model()` 就地修改模型——冻结所有原始权重并挂载 LoRA adapter。

注意应用 LoRA 后参数数量的变化。从 1.25 亿可训练参数降到不到 80 万，这就是效率提升的实际体现。

> 📄 **对应代码文件：** [lora_config_example.py](llm-lora-qlora-finetuning-guide/lora_config_example.py) —— 完整版为 Qwen2-7B 配置 LoRA 的示例（含 ModelScope 本地模型加载）

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,                 # alpha/r = 2.0——标准规则
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print("应用 LoRA 之后:")
print(f"  可训练:  {trainable:,}  ({trainable / total * 100:.2f}%)")
print(f"  冻结:     {total - trainable:,}")
model.print_trainable_parameters()
```

```python
应用 LoRA 之后:
  可训练:  786,432  (0.63%)
  冻结:     125,197,312

trainable params: 786,432 || all params: 125,983,744 || trainable%: 0.6242
```

### 第 4 步——用 SFTTrainer 训练

SFTTrainer 处理指令微调的整套流程：提示词/回答屏蔽、序列打包，以及与 PEFT 的干净集成。

关键训练参数：`num_train_epochs`（1–3 是标准；更多容易过拟合）、`gradient_accumulation_steps`（模拟更大的批次：有效批次 = `batch_size × 该值`）、`learning_rate`（2e-4 是 LoRA 微调可靠的起始点）。

> 📄 **对应代码文件：** [lora_finetune_opt.py](llm-lora-qlora-finetuning-guide/lora_finetune_opt.py) —— OPT-125M 完整 LoRA 微调脚本（含数据加载、训练）

```python
training_args = SFTConfig(
    output_dir="./opt-125m-lora",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # 有效批次大小 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_steps=100,
    logging_steps=10,
    bf16=True,
    max_seq_length=512,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 预期：在 T4 GPU 上训练 1 个 epoch 约 15 分钟
trainer.train()
print("训练完成!")
```

### 第 5 步——保存与评估

LoRA 只保存 adapter 权重——而不是整个模型。对于 rank=16 的 OPT-125M，adapter 文件夹大约 3 MB，而完整模型约 250 MB。

训练后做一个快速健全性检查：比较模型在测试提示词上微调前后的回答。如果输出在你的任务上有明显改进，说明训练生效了。

> 📄 **对应代码文件：** [lora_finetune_opt_save_adapter.py](llm-lora-qlora-finetuning-guide/lora_finetune_opt_save_adapter.py) —— LoRA 微调 + 保存 adapter（约 3 MB）

```python
# 只保存 adapter（很小）
adapter_save_path = "./opt-125m-lora-adapter"
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

adapter_size_mb = sum(
    os.path.getsize(os.path.join(root, f))
    for root, dirs, files in os.walk(adapter_save_path)
    for f in files
) / 1e6
print(f"Adapter 已保存: {adapter_size_mb:.1f} MB  (完整模型约 250 MB)")
```

```python
Adapter 已保存: 3.1 MB  (完整模型约 250 MB)
```

> 📄 **对应代码文件：** [lora_finetune_opt_evaluate.py](llm-lora-qlora-finetuning-guide/lora_finetune_opt_evaluate.py) —— 加载 adapter 并生成回答进行快速评估

```python
# 加载 adapter 并快速评估
base = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
fine_tuned = PeftModel.from_pretrained(base, adapter_save_path)
fine_tuned.eval()

test_prompt = "### Human: Explain the concept of gradient descent.\n### Assistant:"
inputs = tokenizer(test_prompt, return_tensors="pt").to(fine_tuned.device)

with torch.no_grad():
    output_ids = fine_tuned.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
print("回答:", generated)
```

**快速自测：** 为什么我们只解码 `output_ids[0][inputs["input_ids"].shape[1]:]` 而不是完整输出？因为 `generate()` 的输出包含原始提示词——我们把它切掉，只展示新生成的 token。

## QLoRA——在消费级硬件上微调十亿参数模型

LoRA 大幅降低了训练内存——但基础模型本身在 bfloat16 下对 7B 模型仍要占 14 GB。这对大多数消费级 GPU 来说太大了。

QLoRA 将冻结的基础模型压缩为 **4-bit 精度**——从 14 GB 降到约 3.5 GB。LoRA adapter 保持 16-bit 精度。你得到的是 4-bit 的存储、16-bit 的训练质量。

三项技术让它在不牺牲模型质量的前提下实现：

**1. 4-bit NormalFloat（NF4）**
标准的 4-bit 整数量化在整个数值范围内均匀分布量化级别。但神经网络权重聚集在零附近，呈正态分布。NF4 在接近零的地方（大多数权重所在处）分配更多量化级别，在极值处分配更少。同样的 4 个比特能远更准确地表示这种分布。

**2. 双重量化（Double Quantization）**
量化需要"校准常量"来把量化权重映射回真实数值。这些常量本身也占内存。双重量化把这些常量也量化一遍，在整个模型上每个参数节省约 0.37 比特。

**3. 分页优化器（Paged Optimizers）**
当 GPU 内存意外飙升时（长序列、变长批次），NVIDIA 的统一内存会把优化器状态分页到 CPU 内存。这能在最坏的时刻防止内存溢出崩溃——就像 GPU 的虚拟内存。

三者合力，一个 7B 模型在 6 GB 以下就能放下。一个 65B 模型可以放进单张 A100。

> **QLoRA = 4-bit 基础模型 + 16-bit LoRA adapter。** 冻结权重以 4-bit 存储，每次前向计算时反量化回 bfloat16，用完即弃。adapter 的梯度全程以 bfloat16 流动。你得到的是 4-bit 的内存、bfloat16 的训练稳定性。

### 与 LoRA 的唯一区别：加载模型的方式

QLoRA 的配置几乎和 LoRA 完全一样。唯一的改动是在加载模型时加上一个 `BitsAndBytesConfig`——它会在加载时把模型压缩成 4-bit。

加载量化模型后必须额外加两行代码。`use_cache = False` 禁用 KV 缓存，因为它与梯度检查点（gradient checkpointing）冲突。`enable_input_require_grads()` 确保梯度能穿过量化模型流入 LoRA adapter。忘了任何一个都会在训练时产生令人困惑的错误。

> 📄 **对应代码文件：** [qlora_config_example.py](llm-lora-qlora-finetuning-guide/qlora_config_example.py) —— QLoRA 配置 + 4-bit 模型加载

```python
# QLoRA 第 1 步：配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                 # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,     # 计算时反量化为 bfloat16
    bnb_4bit_use_double_quant=True,            # 双重量化：每个参数节省约 0.37 bit
)

# QLoRA 第 2 步：带量化加载——与 LoRA 相比唯一不同的那一行
model_qlora = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 加载量化模型后，这两行是必须的
model_qlora.config.use_cache = False        # 与梯度检查点不兼容
model_qlora.enable_input_require_grads()    # 让梯度能到达 LoRA adapter

print("模型以 4-bit 精度加载")
print(f"模型权重精度: {next(model_qlora.parameters()).dtype}")
```

```python
模型以 4-bit 精度加载
模型权重精度: torch.uint8
```

> 📄 **对应代码文件：** [qlora_finetune_opt.py](llm-lora-qlora-finetuning-guide/qlora_finetune_opt.py) —— QLoRA 完整微调脚本（4-bit 加载 + LoRA + paged_adamw_32bit 训练）

```python
# QLoRA 第 3 步：应用 LoRA——与标准 LoRA 配置完全相同
qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model_qlora = get_peft_model(model_qlora, qlora_config)
model_qlora.print_trainable_parameters()

# QLoRA 第 4 步：训练——与 LoRA 相同，另加 paged_adamw_32bit
qlora_args = SFTConfig(
    output_dir="./opt-125m-qlora",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    max_seq_length=512,
    dataset_text_field="text",
    optim="paged_adamw_32bit",   # 分页优化器——防止训练尖峰时 OOM
    report_to="none",
)

qlora_trainer = SFTTrainer(
    model=model_qlora,
    args=qlora_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
qlora_trainer.train()
print("QLoRA 训练完成!")
```

```python
trainable params: 786,432 || all params: 125,983,744 || trainable%: 0.6242
```

> **用 Flash Attention 2 加速训练。** 如果你的 GPU 支持（Ampere/Hopper 或更新架构），在 `from_pretrained()` 调用中加入 `attn_implementation="flash_attention_2"`。它使用一种内存高效的注意力算法，在长序列上可以把训练时间缩短 2–4 倍，且质量无损。
>
> ```python
> model = AutoModelForCausalLM.from_pretrained(
>     model_name,
>     attn_implementation="flash_attention_2", # 需要：pip install flash-attn
>     torch_dtype=torch.bfloat16,
>     device_map="auto",
> )
> ```

> 📄 **对应代码文件：** [qlora_finetune_opt_flash_attn.py](llm-lora-qlora-finetuning-guide/qlora_finetune_opt_flash_attn.py) —— QLoRA + 高效注意力（SDPA / Flash Attention 2）完整训练脚本

## [动手练习] 练习 2：为 Llama 3.1 8B 配置 QLoRA

你想在客服数据集上微调 `meta-llama/Llama-3.1-8B-Instruct`。目标模块：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。

**你的任务：** 创建一个带双重量化的 4-bit NF4 的 `BitsAndBytesConfig`，以及一个 `r=64`、覆盖全部 7 个模块的 `LoraConfig`。

*提示 1：`bnb_4bit_quant_type="nf4"` 且 `bnb_4bit_use_double_quant=True`。*
*提示 2：如果 r=64，那么 lora_alpha=128（2×r 规则）。*

```python
bnb_config_llama = BitsAndBytesConfig(
    load_in_4bit=___,
    bnb_4bit_quant_type=___,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=___,
)

qlora_config_llama = LoraConfig(
    r=___,
    lora_alpha=___,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=___,
)

print(f"r: {qlora_config_llama.r}, alpha: {qlora_config_llama.lora_alpha}")
print(f"target_modules: {sorted(qlora_config_llama.target_modules)}")

# 答案：
# bnb_config_llama = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
# qlora_config_llama = LoraConfig(r=64, lora_alpha=128, ...,
#     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
```

---

## 将 LoRA Adapter 合并进基础模型

对于生产部署，你可以把 adapter 永久烘焙进基础模型。`merge_and_unload()` 会为每个适配层计算 W* = W + (α / r) · B · A 并存储结果。输出是一个标准的 HuggingFace 模型——没有 PEFT 包装器，没有运行时开销。

有一个限制要知道：你不能直接合并到 QLoRA 模型上。4-bit 基础模型的权重精度不足以胜任合并运算。先把基础模型用 bfloat16 重新加载，然后再合并。

> 📄 **对应代码文件：** [lora_merge_adapter.py](llm-lora-qlora-finetuning-guide/lora_merge_adapter.py) —— 将 LoRA adapter 合并进基础模型

```python
# 以全精度重新加载基础模型
base_for_merge = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

# 叠加上保存好的 adapter
peft_model = PeftModel.from_pretrained(base_for_merge, adapter_save_path)

# 把 adapter 权重烘焙进基础模型 (W* = W + alpha/r * B @ A)
merged_model = peft_model.merge_and_unload()

print(f"合并后模型类型: {type(merged_model).__name__}")
print(f"是否还是 PEFT 模型: {hasattr(merged_model, 'peft_config')}")

# 像任何标准 HuggingFace 模型一样保存
merged_model.save_pretrained("./opt-125m-merged")
tokenizer.save_pretrained("./opt-125m-merged")
print("合并后的模型已保存。")
```

```python
合并后模型类型: OPTForCausalLM
是否还是 PEFT 模型: False
合并后的模型已保存。
```

---

## LoRA vs QLoRA——什么时候用哪个

两者都有效。正确的选择取决于你的硬件。

| 特性 | LoRA | QLoRA |
|---|---|---|
| 基础模型精度 | bfloat16（完整） | 4-bit NF4（量化） |
| 7B 模型训练显存 | 约 18+ GB | 约 6 GB |
| 训练速度 | 基准 | 慢约 30%（反量化开销）[2] |
| 最终模型质量 | 略高 | 在大多数基准上追平 LoRA [3] |
| 能否合并 adapter？ | 可以，直接合并 | 可以，但需先以 bf16 重新加载 |
| 最佳硬件适配 | ≥16 GB 显存 | 6–16 GB 显存 |

**选择指南：**
- 免费 Colab T4（15 GB）+ 7B 模型 → **QLoRA**
- RTX 3080 10 GB + 7B 模型 → **QLoRA**
- RTX 4090 24 GB + 7B 模型 → **LoRA**
- 任何配置 + 13B 模型 → **QLoRA**

> **避免双重量化。** 如果你用 QLoRA 微调、合并、然后再把合并后的模型重新量化为 4-bit 用于部署，就会把两轮有损压缩的量化误差叠加起来。正确做法：用 LoRA 在 bfloat16 下微调、合并，然后用专门的工具（如 llama.cpp/GGUF 格式）一次性量化。

## 什么时候不该用 LoRA 或 QLoRA

LoRA 并不总是正确的工具。有三种场景你需要重新考虑：

**1. 你的数据集覆盖了基础模型从未见过的全新领域。**
LoRA 适配的是已有知识——它放大和重定向模型已经知道的东西。如果你要教模型一个术语和概念完全独特的专业领域，小秩的 adapter 可能容量不足。考虑全参数微调，或者把秩提高到 64–128。

**2. 你需要逐位可复现的模型权重。**
合并 LoRA adapter 会引入浮点舍入，这与完全微调的模型不同。实践中这几乎无法察觉。但在需要精确权重校验和（checksum）的受监管环境中，这一点很重要。

**3. 你的任务需要大幅改写基础模型的行为。**
LoRA 保留了基础模型的大部分行为。如果微调需要从根本上改变模型的行为方式——而不是精修——全参数微调给你更多控制。实践检验：如果 LoRA 的输出里渗透出基础模型的风格，就提高秩，或者切换到全参数微调。

## 常见错误及修复方法

### 错误 1：把 `lora_alpha` 设得太低（结果不对）

❌ **错误：**

```python
LoraConfig(r=16, lora_alpha=1, ...)  # alpha/r = 0.0625
```

**为什么会失败：** 缩放因子 α / r = 1/16 = 0.0625。adapter 的更新小到几乎不影响模型。训练损失确实在下降，但微调后的模型行为与基础模型几乎相同。

✅ **正确：**

```python
LoraConfig(r=16, lora_alpha=32, ...)  # alpha/r = 2.0
```

---

### 错误 2：QLoRA 忘记设 `use_cache = False`（训练崩溃）

❌ **错误：** 加载量化模型时没有禁用 KV 缓存。

**为什么会崩溃：** 梯度检查点与 KV 缓存不兼容。报错信息令人困惑，而且完全看不出根因。

✅ **正确：**

```python
model.config.use_cache = False
model.enable_input_require_grads()
```

---

### 错误 3：只针对 Q 和 V 投影（质量次优）

❌ **次优：**

```python
LoraConfig(target_modules=["q_proj", "v_proj"], ...)
```

在数百次 LoRA 实验 [3] 中发现，针对所有线性层在下游基准上一贯比部分覆盖好 1–3 个百分点。

✅ **更优：**

```python
LoraConfig(target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"], ...)
```

### 错误 4：在小数据集上训练太多轮（过拟合）

❌ **错误：** 在 1,000 个样本的数据集上训练 10 轮。

**为什么会失败：** 模型会精确记忆训练样本。数据集少于 10,000 个样本时，最多用 1–2 轮。盯着评估损失——如果训练损失下降而评估损失上升，立刻停止。

## [动手练习] 练习 3：调试一个配置错误的 QLoRA 配置

下面的代码有**三个 bug**——一个在 `BitsAndBytesConfig` 里，一个在 `LoraConfig` 里，一个在模型加载里。找出并修复全部三个。

*提示 1：检查 compute dtype 以及 alpha/rank 的关系。*
*提示 2：第三个 bug 是缺了一行——想想量化模型加载之后总跟着什么。*

```python
# 有 BUG 的代码——找出三个 bug

bnb_config_buggy = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,  # Bug 1：错误——应该用 bfloat16
    bnb_4bit_use_double_quant=True,
)

lora_config_buggy = LoraConfig(
    r=32,
    lora_alpha=8,           # Bug 2：alpha/r = 0.25——应该是 64 (= 2 × r)
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
)

# model = AutoModelForCausalLM.from_pretrained(...)
# Bug 3：缺少 model.config.use_cache = False 和 enable_input_require_grads()

# Bug 1: bnb_4bit_compute_dtype=torch.bfloat16
# Bug 2: lora_alpha=64  (= 2 × r=32)
# Bug 3: 补上 model.config.use_cache = False 和 model.enable_input_require_grads()
```

---

## 完整代码

点击展开完整脚本（复制粘贴即可运行）

> 📄 **对应代码文件：** [finetune_llm_with_lora_and_qlora.py](llm-lora-qlora-finetuning-guide/finetune_llm_with_lora_and_qlora.py) —— LoRA + QLoRA 端到端微调完整脚本（含 HF 镜像源设置）

```python
# 完整代码：使用 LoRA 和 QLoRA 在 Python 中微调 LLM
# pip install torch transformers peft trl bitsandbytes datasets accelerate
# Python 3.9+ | 需要 6+ GB 显存的 GPU

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL_NAME = "facebook/opt-125m"
ADAPTER_PATH = "./opt-125m-lora-adapter"
MERGED_PATH = "./opt-125m-merged"

# ── LoRA 微调 ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))
model.print_trainable_parameters()

SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        output_dir="./opt-125m-lora", num_train_epochs=1,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, max_seq_length=512,
        dataset_text_field="text", report_to="none",
    ),
).train()
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

# ── 将 adapter 合并进基础模型 ────────────────────────────────
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
PeftModel.from_pretrained(base, ADAPTER_PATH).merge_and_unload().save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)

# ── QLoRA 微调 ────────────────────────────────────────────
model_q = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ),
)
model_q.config.use_cache = False
model_q.enable_input_require_grads()
model_q = get_peft_model(model_q, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))
SFTTrainer(
    model=model_q, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        output_dir="./opt-125m-qlora", num_train_epochs=1,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, max_seq_length=512,
        dataset_text_field="text", optim="paged_adamw_32bit", report_to="none",
    ),
).train()

print("流水线全部完成。")
```

---

## 常见问题解答

### 机器学习中的 LoRA 是什么？

LoRA（Low-Rank Adaptation，低秩适配）是一种面向大语言模型的参数高效微调方法。它在每个冻结的权重层旁边注入两个小的可训练矩阵。训练期间只有这些 adapter 矩阵更新——原始模型权重从不改变。这把可训练参数从数十亿降到数百万，使得在消费级硬件上微调成为可能。

### QLoRA 微调 7B 模型需要多少 GPU 内存？

QLoRA 对一个 70 亿参数的模型大约需要 6 GB GPU 显存。它通过把冻结的基础模型从 14 GB（bfloat16）压缩到约 3.5 GB（4-bit NF4）来实现，同时让 LoRA adapter 保持 16-bit 精度。免费的 Colab T4 GPU（15 GB）就足够了。

### 我可以用 LoRA 微调 BERT 这类仅编码器模型吗？

可以。分类任务设 `task_type=TaskType.SEQ_CLS`，NER 任务设 `TaskType.TOKEN_CLS`。BERT 风格模型的 target module 在 `BertAttention` 里叫 `query`、`value`。LoRA 的数学完全一样——只有任务类型和模块名不同。

### 如何把 LoRA 权重合并进基础模型？

以 bfloat16 精度加载基础模型，用 `PeftModel.from_pretrained()` 把 LoRA adapter 叠上去，然后调用 `.merge_and_unload()`。这把 adapter 公式 W* = W + (α/r) · B · A 烘焙进每一层。你不能直接合并到 4-bit 的 QLoRA 模型上——必须先以 bfloat16 重新加载。

### 训练时 GPU 内存用完了怎么办？

按顺序尝试：(1) 从 LoRA 切换到 QLoRA。(2) 把 `per_device_train_batch_size` 减半，把 `gradient_accumulation_steps` 加倍——有效批次大小不变，内存下降。(3) 减小 `max_seq_length`——内存随序列长度呈二次方增长。(4) 在 SFTConfig 里加 `gradient_checkpointing=True`——节省约 40% 内存，训练慢约 20%。

### 如何选择合适的 LoRA 秩？

从 `r=16` 开始。它对 90% 的微调任务都很好。如果评估损失早早进入平台期、且你有富余显存，再提高到 32 或 64。数据集非常小（少于 1,000 个样本）时降到 4 或 8，以降低过拟合风险。超过 64 的秩很少有帮助，而且总是更费内存。

---

## 接下来做什么

你已经用 LoRA 和 QLoRA 各微调了一个模型。这里有三个自然的下一步：

1. **用自己的数据集微调**——拿一份领域专属的问答 CSV，用上面展示的 `### Human:` / `### Assistant:` 模板格式化，然后用你的数据跑这套流水线。
2. **探索 DPO 微调**——直接偏好优化（Direct Preference Optimization）通过在好/坏回答对上进行训练，让模型对齐人类偏好。它直接建立在你这里学会的 LoRA 配置之上。
3. **加入 Flash Attention 2**——如果你有 Ampere 架构 GPU，`attn_implementation="flash_attention_2"` 参数可以把训练时间缩短 2–4 倍。

---

## 参考资料

1. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. [链接](https://arxiv.org/abs/2106.09685)
2. Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314. [链接](https://arxiv.org/abs/2305.14314)
3. Biderman, D., et al. (2024). *LoRA Learns Less and Forgets Less. Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments.* Lightning AI. [链接](https://lightning.ai/pages/community/lora-insights/)
4. Dettmers, T., et al. (2023). 关于目标模块的消融实验：QLoRA 论文表 6. [链接](https://arxiv.org/abs/2305.14314)
5. Hugging Face PEFT 库——官方文档. [链接](https://huggingface.co/docs/peft/index)
6. Hugging Face TRL 库——SFTTrainer 文档. [链接](https://huggingface.co/docs/trl/sft_trainer)
7. bitsandbytes——4-bit 量化文档. [链接](https://huggingface.co/docs/bitsandbytes/main/en/index)
8. Raschka, S. (2023). *Practical Tips for Finetuning LLMs Using LoRA.* [链接](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
