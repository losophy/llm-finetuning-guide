# Unsloth 微调实战指南

> **运行方式**：训练与导出全部在云端 GPU 完成（**本文用 Kaggle 免费额度**；Google Colab 亦可，见 1.1），本地不装 CUDA、不下模型权重、不占磁盘；只在最后一步用**本地已装的 Ollama** 运行导出的模型。
>
> **学习目标**：掌握 Unsloth 微调框架，能独立完成模型的微调、参数调优、GGUF 导出与本地部署
>
> **前置知识**：已掌握 LoRA/QLoRA 基础理论（参见 `LoRA与QLoRA微调大语言模型完整指南.md`）

---

## 怎么用这份指南

**先跑通，再理解。** Unsloth 官方明确建议初学者从预置 Notebook 起步——先把整条链路完整跑一遍，再回头抠细节，比先啃概念快得多。

**这份指南不是"六关依次闯"。** 真正要跟着做的只有三件事，其余章节是备查资料，按需翻阅即可：

| 顺序 | 做什么 | 看哪节 | 产出 |
|------|--------|--------|------|
| **第 0 步** | 跑一个官方 Notebook，**一行代码都不改** | 本节下方 | 建立"完整流程长什么样"的全局印象 |
| **第 1 步** | 用 Qwen2.5-7B + 默认公开数据集（**自动下载，无需自备**）跑通一次 QLoRA 微调 | 四、微调实战 | LoRA adapter |
| **第 2 步** | 导出 GGUF → 下载到本地 → Ollama 运行 | 六、导出与部署 | 本地可对话的模型 |

其余章节的用法：

| 章节 | 用法 | 什么时候看 |
|------|------|-----------|
| 一、云端平台与显存门槛 | 决策参考 | 开局扫一眼，确认平台与配置数字 |
| 二、云端环境准备 | 备查 | 仅当你自己写脚本（不用官方 Notebook）时才需要 |
| 三、数据集准备 | 知识 / 工具 | 默认会自动下载公开数据集、无需准备；要换自己的数据时，看 3.1 格式与 3.2 脚本 |
| 四、微调实战 | **跟着做** | 第 1 步动手时 |
| 五、参数调优与排错 | 备查 | 训练报错、显存不足、效果差时再翻 |
| 六、导出与部署 | **跟着做** | 第 2 步动手时 |

### 第 0 步：先跑一个官方 Notebook

打开下面任一链接（点开就是可运行 Notebook），**不改任何代码**，按顺序把单元格跑完：

- **Llama 3.1 (8B) Alpaca**（入门首选）
  https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_%288B%29-Alpaca.ipynb
- **Llama 3.2 (1B + 3B) 对话**（模型更小，跑得更快）
  https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_%281B_and_3B%29-Conversational.ipynb

跑完后你会看到：**加载模型 → 格式化数据 → 训练 → 推理测试**，这四件事在一份 Notebook 里全干完了。后面第 1、2 步就是把其中"训练"和"导出"换成 Qwen2.5-7B 与本文的默认数据集——**数据会自动下载，不用自己准备**。

---

## 一、云端平台与显存门槛

> **用法**：决策参考——开局扫一眼、确认平台与配置数字即可，不用跟做。

### 1.1 Colab 与 Kaggle 怎么选

| | Kaggle（首选） | Google Colab（备选） |
|---|---|---|
| GPU | 2×T4（≈30GB，默认只用单卡） | 单 T4（≈15GB） |
| 免费额度 | 每周 30 小时，更可预期 | 动态限时、不保证；耗尽会连不上 GPU |
| 持久化 | 写入 `/kaggle/working`（会话结束自动保存为 Output）或存成 Dataset | 挂载 Google Drive |
| 联网 | **需先验证账号**才能同时开 GPU 和 Internet | 默认可用 |
| 适合 | 长期跑、导出 GGUF（**本文主线**） | 额度恢复后快速起步、跟官方 Notebook |

**推荐路径**：本文主线走 **Kaggle**——额度稳定（每周 30 小时、可预期）、导出 GGUF 时的系统内存也更宽裕。Colab 免费额度是**动态限时、不保证**的（额度耗尽会连不上 GPU 后端），适合额度恢复后点开官方 Notebook 快速起步。

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

| 模型 | 平台 | batch_size | grad_accum | max_seq_len | r | 说明 |
|------|------|------------|------------|-------------|---|------|
| 3B QLoRA | Kaggle T4 | 4 | 2 | 2048 | 16 | 从容运行 |
| **7B QLoRA** | **Kaggle T4** | **2** | **4** | **2048** | **16** | **本文默认配置** |
| 8B QLoRA | Kaggle T4 | 2 | 4 | 2048 | 16 | Llama 3.1 等 |
| 14B QLoRA | Kaggle 2×T4 | 2 | 4 | 2048 | 16 | 需更大显存 |
| 7B LoRA(16-bit) | Kaggle 2×T4 | 1 | 8 | 2048 | 16 | 19GB 显存需求 |

> 前三行的 `Kaggle T4` 指**单张 T4（≈15GB）**，与 Colab 的 T4 配置完全相同，换平台不用改任何超参。Kaggle 默认给 T4 ×2，会话开头用 `%env CUDA_VISIBLE_DEVICES=0` 限定单卡（见 2.1）。

> 这张表是全篇的配置总表（第五章不再重复列出）。这些值实际写在第四章代码的 `SFTConfig` 与 `get_peft_model` 里。

---

## 二、云端环境准备

> **用法**：备查——用官方 Notebook 时可跳过（Notebook 第一格会自动安装 Unsloth）。只有你自己写脚本时才需要这一章。

### 2.1 环境配置（Colab / Kaggle 通用）

**步骤 1：选好 GPU 运行时**

- **Kaggle（本文主线）**：右侧设置面板里 Accelerator 选 `GPU T4 ×2`，并把 **Internet 打开**（两者都要求账号已验证）。注意 Kaggle 的 GPU 档位**只有 `GPU T4 ×2`**（没有"单 T4"可选），两张卡都会分配给你——本文只用一张，下一步立刻限掉。
- **Colab（备选）**：菜单 代码执行程序 → 更改运行时类型 → T4 GPU。Colab 本来就只有一张卡，**跳过步骤 2**。

**步骤 2：限定单卡（仅 Kaggle 需要）**

Kaggle 分给你 2 张 T4。不限制的话，Unsloth 检测到两张卡会**自动把模型切开、跨卡训练**（张量并行）——既慢，又和本文"单卡 T4"的配置口径对不上。所以**第一个代码格**就执行：

```python
%env CUDA_VISIBLE_DEVICES=0
```

> ⚠️ **这一格必须排在本 notebook 任何 `import torch` / `torch.cuda.*` 之前——也就是整份 notebook 的第一格。**
> `CUDA_VISIBLE_DEVICES` 只在 CUDA **首次初始化**时生效。一旦跑过下面的步骤 3、或任何加载模型的代码，CUDA 就已经在两张卡上初始化完毕，此时再设**无效**，模型照旧吃满两张卡。
> **怎么确认生效**：步骤 3 会打印"可见 GPU 数"（应为 1）；加载模型时 Unsloth 横幅应显示 `Num GPUs = 1`。若看到 `Num GPUs = 2` 或 `output head -> cuda:1`，说明限晚了——CUDA 初始化不可逆，只能**重启会话**重来。

**步骤 3：确认 GPU 可用**

```python
# 两个平台都跑这一格
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"可见 GPU 数: {torch.cuda.device_count()}")  # Kaggle 限卡后应为 1
```

**步骤 4：安装 Unsloth**

取自 Unsloth 官方 notebook 的安装格，**去掉了 if/else 分支**（原因见下），Colab / Kaggle 通用，版本已钉死在 Unsloth 测试过的组合上：

```python
import re
import torch

v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")

!pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
!pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
!pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
```

**步骤 5：确定工作目录（成果往哪放）**

- **Kaggle**：直接写 `/kaggle/working/`，**不需要 mount**——会话结束时它自动保存为 Output，下次新建 Notebook 可把它挂成输入数据。（⚠️ 一个例外：导出 GGUF 时 Unsloth 可能因空间不足**自动改写到 `/tmp`**，那不存 Output，详见 6.2 / 6.3 第一步。）
- **Colab**：先挂载 Google Drive（见下面的代码块）。

```python
# 仅 Colab 需要
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/unsloth-project
```

### 2.2 平台差异与注意事项

| | Kaggle（首选） | Colab（备选） |
|---|---|---|
| 开 GPU | Accelerator 选 `GPU T4 ×2` | 代码执行程序 → 更改运行时类型 → T4 GPU |
| 联网 | **必须手动打开 Internet**（且账号已验证） | 默认可用 |
| 单卡限定 | **必须**，且要放在首个代码格（见 2.1 步骤 2） | 本来就是单卡 |
| 成果存放 | `/kaggle/working/`（自动存为 Output） | 挂载 Google Drive（见 2.1 步骤 5） |
| 额度 | 每周 30 小时，更可预期 | 动态限时、不保证；耗尽会连不上 GPU |
| 系统内存 | 宽裕，7B 导出基本能过 | 较紧——**导出 GGUF 时可能不足** |

> **为什么主线走 Kaggle**：Colab 的免费额度是**动态限时、不保证**的——额度耗尽时新建 Notebook 直接连不上 GPU 后端（本文这次切换就是这个原因）。Kaggle 每周 30 小时、更可预期，导出 GGUF 时系统内存也更宽裕。**Colab 额度恢复后仍可换回去**（它的好处是点开即用），代码一行不用改。
> **Kaggle 的两个硬前提**：① **账号需完成验证**（手机号），否则 GPU 与 Internet 不能同时开；② **Internet 必须打开**，否则 `pip` 装不了、数据集也下不来。
> **为什么限定单卡**：本文的配置与显存口径都是**按单张 T4** 算的。Kaggle 只会给 `GPU T4 ×2`（没有单卡档位），不限制的话 Unsloth 会把模型切开跨两张卡跑、还多吃系统内存与额度。**实测印证**：同一份配置，双卡张量并行每步 5.6 s、限成单卡后 5.7 s——跨卡通信把第二张卡的收益基本吃光了，等于白占一份额度。**关键是时序**：`%env CUDA_VISIBLE_DEVICES=0` 必须在 CUDA 初始化之前——否则无效，只能重启会话（详见 2.1 步骤 2）。

### 2.3 存储与断线续训

免费额度随时会断，训练成果必须落在持久化存储里：

| | Kaggle（首选） | Colab（备选） |
|---|---|---|
| 放哪 | `/kaggle/working/`（会话结束时自动保存为 Output） | `/content/drive/MyDrive/unsloth-project`（需先 `drive.mount()`） |
| 重连后 | 把 Output 挂载为新 Notebook 的输入数据 | 重新 mount 即可访问 |

**控制 checkpoint 体积**（两个平台都适用，免费空间都不大）：

- **只保留最新一个 checkpoint**：训练参数里加 `save_total_limit=1`
- **训练完只下载 adapter**（几十 MB），不要下载整个 checkpoint 目录
- Colab 免费 Drive 只有 15GB，7B 的 checkpoint（含优化器状态）单个可能上 GB，多个叠加很快会满

### 2.4 验证环境

检查 PyTorch / CUDA / GPU / VRAM 是否正常：

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"VRAM 容量: {gpu_props.total_memory / 1e9:.1f} GB")
    print(f"计算能力: {gpu_props.major}.{gpu_props.minor}")
    # 判据用「算力」，不要用 torch.cuda.is_bf16_supported()——原因见下方说明
    print(f"BF16 原生支持: {gpu_props.major >= 8}")
else:
    print("警告: 未检测到 NVIDIA GPU，请确保已启用 GPU 运行时")
```

> **预期结果**：CUDA 可用 = `True`，VRAM ≈ 15.x GB（T4 / 16GB 卡），**BF16 原生支持 = `False`**——T4 的算力是 7.5，**没有原生 bf16**，所以本项目全程用 **fp16**（见 5.1）。这一行就是提前发现"精度跑错"的信号灯。
>
> ⚠️ **判据必须是「算力」，不能用 `torch.cuda.is_bf16_supported()`**。新版 PyTorch 的这个函数默认把"软件模拟"也算作支持（`including_emulation` 默认为 `True`）——**在 T4 上它会返回 `True`**（本项目实测 torch 2.11.0+cu128 = `True`），但那是 fp32 模拟出来的，**并非真能用**：社区在 Kaggle T4 上的同类任务实测，bf16 模拟比 fp16 慢约 4×。
>
> 对照着看：`torch.cuda.is_bf16_supported(including_emulation=False)` 才会返回 `False`。为避免依赖这个参数名，4.1 / 5.1 的精度开关统一写成 `torch.cuda.get_device_capability()[0] >= 8`。

---

## 三、数据集准备

> **用法**：知识 / 工具——**默认路径无需准备数据**：3.2 会自动下载公开数据集，直接跑。3.1 的两种格式要懂，等你要换成自己的数据时再回来看。

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

> **数据集来源**：本指南不附带数据文件。默认加载 HuggingFace Hub 的公开数据集 `yahma/alpaca-cleaned`（Alpaca 格式），可换成中文数据集（如 `shibing624/alpaca-zh`），也可传入本地 JSONL——下面的代码会自动识别 Alpaca / ShareGPT 两种字段。
> 中国大陆网络可先设镜像：`export HF_ENDPOINT=https://hf-mirror.com`

```python
from datasets import load_dataset

# 默认数据集（HuggingFace Hub，Alpaca 格式）
# 英文通用：yahma/alpaca-cleaned   中文指令：shibing624/alpaca-zh
DEFAULT_DATASET = "yahma/alpaca-cleaned"
MAX_SAMPLES = 1000          # 先小规模跑通，确认无误后再调大（"100 条精品"原则）


def load_from_hub(dataset_name=DEFAULT_DATASET, max_samples=MAX_SAMPLES):
    """从 HuggingFace Hub 加载公开数据集（默认 Alpaca 格式）"""
    dataset = load_dataset(dataset_name, split="train")
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    return dataset


def convert_sharegpt(sample):
    """把 ShareGPT 多轮对话（role/content 或 from/value）转成 Alpaca 三字段"""
    def pick(roles):
        for turn in sample["conversations"]:
            role = turn.get("role") or turn.get("from")
            if role in roles:
                return turn.get("content") or turn.get("value") or ""
        return ""

    return {
        "instruction": pick(("user", "human")),
        "input": "",
        "output": pick(("assistant", "gpt")),
    }


def load_from_jsonl(path):
    """从本地 JSONL 加载，自动识别 Alpaca / ShareGPT 两种格式"""
    dataset = load_dataset("json", data_files=path, split="train")
    columns = dataset.column_names
    if "instruction" in columns and "output" in columns:
        return dataset            # 已是 Alpaca 格式
    if "conversations" in columns:
        return dataset.map(convert_sharegpt, remove_columns=columns)
    raise ValueError(f"无法识别的数据格式，字段为: {columns}")


def prepare_datasets(source=None, test_size=0.2, seed=42):
    """准备训练集/验证集

    source=None  → 加载 HuggingFace 默认数据集（需要联网）
    source=路径  → 加载本地 JSONL（Alpaca 或 ShareGPT 格式均可）
    """
    dataset = load_from_jsonl(source) if source else load_from_hub()
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


# 默认加载 HuggingFace 数据集；也可传本地文件：prepare_datasets("data/xxx.jsonl")
train_dataset, eval_dataset = prepare_datasets()
```

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

> **用法**：跟着做——第 1 步动手时看这一章。

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
            num_train_epochs=2,                 # 默认 1000 条子集下 1~2 轮即可（3 轮会过拟合）
            learning_rate=2e-4,
            # 精度：按「算力」判断。T4（算力 7.5）没有原生 bf16 → 走 fp16。
            # 别用 torch.cuda.is_bf16_supported()：新版 PyTorch 默认把软件模拟也算作支持，T4 上会误报 True（见 2.4）。
            fp16=torch.cuda.get_device_capability()[0] < 8,
            bf16=torch.cuda.get_device_capability()[0] >= 8,
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
    # 数据准备直接用 3.2 的 prepare_datasets()（那段代码要在同一运行时里先跑过）
    # 保存 adapter 见 6.1；导出 GGUF 见 6.2

    # 加载模型
    model, tokenizer = load_model()

    # 配置 LoRA
    model = configure_lora(model)

    # 准备数据集（prepare_datasets 来自 3.2）
    train_dataset, eval_dataset = prepare_datasets()
    train_dataset = train_dataset.map(lambda x: {"text": format_prompt(x)})
    eval_dataset = eval_dataset.map(lambda x: {"text": format_prompt(x)})

    # 创建训练器
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset)

    # 开始训练
    train(trainer)

    print("训练完成，模型已保存到 ./qwen2.5-finetuned-final")
```

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

> **用法**：备查——不用通读，训练报错、显存不足、效果差时再翻对应条目。

### 5.1 关键参数说明

```python
import torch
from transformers import TrainingArguments

# ── 标准配置（≥16GB 显存，示例：T4）──────────────────────────────
training_args = TrainingArguments(
    # 批次大小
    per_device_train_batch_size=2,      # 根据 VRAM 调整
    gradient_accumulation_steps=4,      # 模拟更大批次（有效批次=2*4=8）

    # 学习率
    learning_rate=2e-4,                 # LoRA 标准起点
    lr_scheduler_type="cosine",         # 余弦退火
    warmup_steps=10,                    # 预热步数（新版已移除 warmup_ratio）

    # 训练轮次
    num_train_epochs=2,                 # 1~2 轮足够；默认 1000 条子集下 3 轮即过拟合

    # 精度：按「算力」判断。T4（7.5）没有原生 bf16，写死 bf16=True 会走 fp32 模拟、静默慢数倍。
    # 别用 torch.cuda.is_bf16_supported()：新版 PyTorch 默认把软件模拟也算作支持，T4 上会误报 True（见 2.4）。
    fp16=torch.cuda.get_device_capability()[0] < 8,
    bf16=torch.cuda.get_device_capability()[0] >= 8,

    # 其他
    gradient_checkpointing=True,        # 节省 VRAM
    optim="adamw_8bit",                 # 8-bit 优化器

    output_dir="./qwen2.5-finetuned",
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
)

# ── 8GB 显存配置（示例：本机 RTX 4060 Ti）────────────────────────
training_args_8gb = TrainingArguments(
    per_device_train_batch_size=1,      # 8GB 必须设为 1
    gradient_accumulation_steps=8,      # 等效批量仍是 8
    # 注意：序列长度不是 TrainingArguments 参数，
    # 请在 FastLanguageModel.from_pretrained(max_seq_length=1024) 处设置

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    num_train_epochs=2,

    fp16=True,                          # 8GB 卡（如 RTX 4060 Ti）用 fp16
    bf16=False,

    gradient_checkpointing=True,
    optim="adamw_8bit",

    output_dir="./qwen2.5-finetuned",
    logging_steps=10,
    save_steps=100,
)
```

> 各硬件档位（batch_size / grad_accum / r / max_seq_length）见 **1.3 推荐配置**（全篇唯一一份配置总表，此处不再重复）。
> 说明：上面用 `TrainingArguments` 讲参数；**实际训练用的是 `SFTConfig`**（TRL 提供，见 4.1），两者参数名完全相同。

### 5.2 调优策略

#### 问题 1：过拟合（训练损失下降，验证损失上升）

> **实测（Kaggle 单卡 T4，同一份 1000 条子集 / train 800，每 100 步评估一次）**
>
> | 轮次 | step 100 | step 200 | step 300 | val 涨幅 |
> |---|---|---|---|---|
> | **3 轮**（旧默认） | train 0.977 / val 1.051 | train 0.834 / val 1.101 | train 0.535 / val **1.226** | **+0.175** |
> | **2 轮**（现默认） | train 0.977 / val 1.050 | train 0.863 / val **1.077** | — | **+0.027** |
>
> 两次的验证损失都持续回升：**3 轮明显过头；改 2 轮后涨幅降到约 1/6**，但没有归零——严格说仍在过拟合。另一处要注意：**保存下来的是最后一步（200）**，而两次评估里验证损失最低的是第 **100** 步。
> 想再压，两条路：① 降到 1 轮（见下面第 2 条建议）；② 保持 2 轮，只让保存取验证损失最低的那一步——加 `load_best_model_at_end=True` + `metric_for_best_model="eval_loss"`（要求 `save_steps` 能整除 `eval_steps`，本文都是 100，已满足）。

- 增加 `lora_dropout`（0.05→0.1）
- **减少 `num_train_epochs`**（数据越少、轮数越要少；本文默认 2，仍过拟合就降到 1）
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
- 看到 `Unsloth: Will smartly offload gradients to save VRAM!` **不是错误**：这是 Unsloth 自动开了「卸载式梯度检查点」，把反向传播要用的中间激活异步挪到系统内存、用完再取回。16GB 卡跑 7B@2048 属于正常策略，官方称额外开销约 2%

#### 问题 4：训练速度慢

> **正常速度参照（实测）**：Kaggle **单卡** T4 + Qwen2.5-7B(4-bit) + `max_seq_length=2048` + batch 2×4 ≈ **5.7 s/step**（200 步 ≈ 19 分钟；此时 Unsloth 已自动开启问题 3 那条梯度卸载）。明显慢于这个量级，再往下查。

- **先查精度档**：T4 上必须真正走 **fp16**。若精度开关是按 `torch.cuda.is_bf16_supported()` 写的，T4 会被误判成"支持 bf16"，实际是 fp32 模拟、**慢数倍**（见 2.4）
- 确认 `use_gradient_checkpointing="unsloth"`（这是 Unsloth 自带的优化内核，**不需要手动安装 Flash Attention 2**——官方不建议在 Windows 上手动编译 FA2）
- 检查 GPU 利用率（`nvidia-smi`）
- 缩短 `max_seq_length`、减小数据集规模

#### 问题 5：模型输出质量差

- 检查数据质量（最重要！）
- 调整 LoRA rank（16→32→64）
- 调整学习率（2e-4→1e-4→5e-5）
- 增加训练轮次

#### 问题 6：如何选择模型？

官方建议**初学者优先用 Instruct 模型**：走对话模板（ChatML / ShareGPT），**需要的数据更少**；Base 模型走 Alpaca 格式，需要更多数据。两条路都能跑通，按你的数据量选：

- **数据量小（100~500 条）**：`unsloth/Qwen2.5-7B-Instruct-bnb-4bit` + ShareGPT 格式
- **数据量充足（数千条）**：`unsloth/Qwen2.5-7B-bnb-4bit` + Alpaca 格式（**本文主线**）
- **高质量需求**：`unsloth/Qwen2.5-14B-bnb-4bit`

### 5.3 配置参数在哪查

各平台 / 模型对应的 batch_size、grad_accum、max_seq_len、r 见 **1.3 推荐配置**（全篇唯一一份配置总表，此处不再重复）。

---

## 六、导出与部署

> **用法**：跟着做——第 2 步动手时看这一章。

> **分工**：云端负责**导出**，本地负责**运行**。云端**不需要安装 Ollama**——GGUF 的生成由 Unsloth 完成，Ollama 只是本地用来加载它的工具。

### 6.1 保存 LoRA adapter（云端）

```python
# 保存 adapter（小文件，通常几十 MB）
model.save_pretrained("./qwen2.5-lora-adapter")
tokenizer.save_pretrained("./qwen2.5-lora-adapter")

# 可选：推送到 HuggingFace
# model.push_to_hub("your-username/qwen2.5-finetuned")
```

下次要用这个 adapter 时，加载同一个基座再挂上去即可：

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",   # 必须是同一个基座
    load_in_4bit=True,
)
model.load_adapter("./qwen2.5-lora-adapter")    # 挂上你训好的 adapter
```

### 6.2 导出为 GGUF（云端）

```python
# 保存为 GGUF 格式
model.save_pretrained_gguf(
    "./qwen2.5-gguf",
    tokenizer,
    quantization_method="q4_k_m"  # Unsloth 默认，本项目选用
)
```

产物文件名**以模型名为前缀**：`./qwen2.5-gguf/Qwen2.5-7B.Q4_K_M.gguf`（**不是** `unsloth.Q4_K_M.gguf`——新版 Unsloth 用模型名做前缀，下载和 Modelfile 里都按实际文件名写）。

**量化方式对照（7B 模型）**：

| 量化方式 | 体积 | 说明 |
|----------|------|------|
| `q4_k_m` | ≈4.7 GB | **本项目选用**（Unsloth 默认），质量与体积平衡 |
| `q5_k_m` | ≈5.4 GB | 质量更高 |
| `q6_k` | ≈6.6 GB | 较高质量 |
| `q8_0` | ≈8 GB | 接近原精度 |
| `q3_k_m` | ≈3.5 GB | 更省空间，质量有损（磁盘实在不够时的退路） |

> 体积口径：上表按 `10³` 换算（`4.7 GB` = `4.36 GiB`）。Windows 资源管理器按 `2³⁰` 显示，你会看到 **"4.36 GB"**——是同一个文件，别以为下错了。

> ⚠️ **真正的瓶颈是磁盘，不是内存。** 导出时同一个目录里要同时放三份东西：合并后的 **16-bit 权重（≈15GB）+ f16 GGUF（≈15GB）+ 最终的 q4（≈4.7GB）**，峰值 **30GB 以上**。而 Kaggle 的 `/kaggle/working/` **只有约 18.7GB 空闲**，装不下 → Unsloth 会**自动把产物改落到 `/tmp`**（Kaggle 的 `/tmp` 有 1TB 空间）并打印：
> `Unsloth: Kaggle's working directory only has 18.7GB free, so the GGUF export goes to /tmp/unsloth_saves/... instead`
> **`/tmp` 是临时空间，会话一停就回收、不会存成 Output**。所以导出成功后**必须先把它搬回 `/kaggle/working/`**（见 6.3 第一步），否则关掉会话成果就没了。这一步与量化档位无关——**降到 `q3_k_m` 也解决不了**（大头是那两份 15GB 中间产物）。
> 若 `q4_k_m` 本身报错，再降到 **`q3_k_m`**（只改上面代码块那一行）。

> 💡 导出时 Unsloth 会自动装 **预编译版 llama.cpp**（`skipping compilation`，省掉约 3 分钟编译），并在转换完成后**自动清理** f16 中间件，最终目录里只剩 q4 那一个文件。

### 6.3 下载到本地

**先纠正一个前提**：Kaggle 上 Unsloth 会把 GGUF 落到 `/tmp`，而 **`/tmp` 不是 Output、会话结束即回收**（原因见 6.2）。所以下载分两步：**先搬进 `/kaggle/working/` 钉成 Output，再想办法弄到本地。**

```python
# Kaggle：把 /tmp 里的最终产物搬回 Output（路径以 6.2 的实际输出为准）
!mkdir -p /kaggle/working/gguf
!cp /tmp/unsloth_saves/qwen2.5-gguf_gguf/Qwen2.5-7B.Q4_K_M.gguf /kaggle/working/gguf/
!ls -lh /kaggle/working/gguf/ && df -h /kaggle/working
```

然后右上角 **Save Version → Quick Save**（⚠️ **别选 "Save & Run All"**，那会把整个 notebook 从头重跑一遍，白等半小时），跑完 Output 里就有这个文件了——**此后会话随便关**。

> Colab 则相反：它的磁盘没有这个限制，正常写盘后挂到 Google Drive 长期保存即可。

#### 三种下载方式

| 方式 | 可靠性 | 说明 |
|---|---|---|
| **HuggingFace 中转** | ★★★ **推荐** | 一次投入，链接**永久不过期**、可续传，以后反复拉模型都能用 |
| 浏览器 / `FileLink` | ★★ | 偶尔下一次可以，但慢、易断，断了只能从头来 |
| 外部下载器（FDM / IDM / aria2） | ⚠️ 对 Kaggle 直链常失败 | Kaggle 链接依赖会话登录态，外部工具通常拿不到；**若报 403 / 401 就是这个原因**。想用多线程续传，请改用下面的 **HF 永久链接** |

#### HuggingFace 中转完整流程

**① 云端上传**（把**已有**的 GGUF 传上去，**不重新导出**）

前提：HF 账号 + 一个 **`Write` 类型**的 access token（**`Read` 类型上传会 403**，这是第一大坑）。

```python
!pip install -q -U huggingface_hub

from huggingface_hub import login, HfApi

login()   # 设备授权：浏览器打开 https://hf.co/oauth/device 并输入提示的验证码

api = HfApi()
api.create_repo("your-username/qwen2.5-7b-gguf", repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="/kaggle/working/gguf/Qwen2.5-7B.Q4_K_M.gguf",
    path_in_repo="Qwen2.5-7B.Q4_K_M.gguf",
    repo_id="your-username/qwen2.5-7b-gguf",
)
```

> ⚠️ **别用 `model.push_to_hub_gguf()`**：它会把整个导出（合并 16-bit → 转 f16 → 转 q4）**重跑一遍**，白等 16 分钟。上面用 `upload_file` 直接传已生成的文件（实测 4.68GB 上传约 1 分钟）。
> ⚠️ `pip install -U huggingface_hub` 可能把 hub 升到 2.x，与镜像自带的 transformers / datasets 报依赖冲突警告——**纯上传不受影响**，但这个会话别再用来跑训练/推理。
> 💡 纯上传**不需要 GPU、不需要 Unsloth**：新建 Notebook 时 Accelerator 选 `None`，用 Add Input 把 Output 挂进来（路径变成 `/kaggle/input/<名称>/gguf/...`，用 `!find /kaggle/input -name "*.gguf"` 查确切路径）。

**② 本地下载**

```bash
pip install -U "huggingface_hub[hf_transfer]"

# PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
#             $env:HF_HUB_ENABLE_HF_TRANSFER="1"    （CMD 用 set HF_ENDPOINT=...）
hf download your-username/qwen2.5-7b-gguf --local-dir D:/models
# 旧版命令：huggingface-cli download（参数相同）
```

**断了就重跑同一条命令**，会跳过已下好的部分接着下。

> **也可以喂给外部下载器**（多线程更猛），但要用 HF 的链接而不是 Kaggle 的：
> `https://hf-mirror.com/your-username/qwen2.5-7b-gguf/resolve/main/Qwen2.5-7B.Q4_K_M.gguf`
> 这是**永久链接、不过期**。前提：仓库要**公开**；私有仓库需先配 token。

**③ 核对**：本地文件应约 **4.68 GB**（= **4.36 GiB**，Windows 资源管理器按 2³⁰ 显示，看到 "4.36 GB" 是对的）。

### 6.4 本地使用 Ollama 运行

前提：本地已安装 Ollama。创建 `Modelfile`（`FROM` 路径按实际存放位置填写）：

```bash
cat > Modelfile << 'EOF'
FROM D:/models/Qwen2.5-7B.Q4_K_M.gguf

TEMPLATE """### Instruction:
{{ .Prompt }}

### Response:
"""

PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
EOF

# 创建 Ollama 模型
ollama create qwen2.5-finetuned -f Modelfile

# 运行模型
ollama run qwen2.5-finetuned
```

> ⚠️ **`TEMPLATE` 和 `STOP` 都不能省，这是最容易翻车的地方。** 训练用的是 **Alpaca** 格式（`### Instruction:` / `### Response:`，见 3.2），而 GGUF 里**内嵌的默认模板是 Qwen2.5 自带的 ChatML**（`<|im_start|>`）。不在 Modelfile 里显式指定 `TEMPLATE`，Ollama 就会拿 ChatML 去问一个只见过 Alpaca 的模型——**训练/推理格式对不上，输出会答非所问**。
> `STOP` 那两条也必须加：Alpaca 格式的模型不加停止词会**自己往下编下一轮 `### Instruction:`**。
> **不要加 `SYSTEM`**：训练数据里没有 system 段，硬塞一句中文系统提示属于训练分布外。
>
> 💡 还有一个相关现象：Unsloth 导出时可能打印 `No Ollama template mapping found for model 'unsloth/Qwen2.5-7B'. Skipping Ollama Modelfile`——它**不会**自动生成 Modelfile。这不是错误，用上面这份手写的即可（base 模型的映射表里没有 Qwen2.5-7B 条目，所以它跳过）。

验证模板是否生效：**英文**提问（训练数据 `yahma/alpaca-cleaned` 是英文，中文属分布外），并留意输出**不会**自行续写下一轮 `### Instruction:`——出现续写就说明 `STOP` 没生效。

> 💡 **省空间要点**：`ollama create` 会把 GGUF **复制**进 Ollama 自己的模型库（Windows 下在 `C:\Users\<用户名>\.ollama\models`），不是引用原文件。
> 所以刚导入完时本地会有**两份**（合计约 9.4GB）。**导入成功后，原始的那个 `.gguf` 就可以删掉**，最终只占约 4.7GB。

### 6.5 云端 ↔ 本地分工一览

| 环节 | 在哪做 | 需要什么 |
|------|--------|----------|
| 环境准备 / 训练 / 调参 | 云端（本文用 Kaggle） | 免费 GPU 额度 |
| 导出 adapter / GGUF | 云端 | Unsloth（**不需要 Ollama**） |
| 从 `/tmp` 搬回 Output（Kaggle 特有） | 云端 | 见 6.3 第一步 |
| 下载 GGUF | 本地 | 约 4.7GB 磁盘空间 |
| 加载并对话 | 本地 | 已安装的 Ollama |

---

## 七、附录：官方 Notebook 清单（250+）

> 下表给的是 Colab 打开链接；本文主线用 **Kaggle**，可 `File → Import Notebook → GitHub` 把同一份 Notebook 导进来跑（把链接里的 `colab.research.google.com/github/` 换成 `github.com/`）。完整列表（250+）：https://github.com/unslothai/notebooks
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

## 完成检查清单（验收用，不是待办）

- [ ] 跑通至少 1 个官方 Notebook（如 Llama 3.1 8B Alpaca）
- [ ] 云端环境可用（Kaggle：Unsloth 装好、Internet 已打开、**首格已执行 `%env CUDA_VISIBLE_DEVICES=0`**）
- [ ] 精度档正确（2.4 打印 **BF16 原生支持 = `False`**，训练实际走 fp16；理由见 2.4 的 ⚠️ 说明）
- [ ] 数据就绪（默认自动下载 `yahma/alpaca-cleaned`，**无需自备**；或换成 Alpaca / ShareGPT 格式的自己的数据）
- [ ] 完成 Qwen2.5-7B 的 QLoRA 微调，产出 LoRA adapter
- [ ] 记录每次实验的参数与 loss
- [ ] 导出 GGUF（q4_k_m；核对产物路径——若落在 `/tmp` 需先搬回 `/kaggle/working/` 再 Save Version，见 6.2 / 6.3）
- [ ] 下载到本地，用 Ollama 跑起来（Modelfile 必带 Alpaca 的 `TEMPLATE` + 两条 `STOP`，见 6.4）
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
