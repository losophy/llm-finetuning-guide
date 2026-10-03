# Unsloth 与原生 HuggingFace 对照

> **这份文档解决什么问题**：你手写过原生 HuggingFace/PEFT 的 LoRA/QLoRA 全流程，也已经跑通了 Unsloth。这份文档只回答两件事——**Unsloth 到底替掉了哪些代码**、**它相对原生 HF 能快多少**。
>
> 它**不教微调**。训练步骤见 `Unsloth微调实战指南.md` 第四章；原理推导见 `LoRA与QLoRA微调大语言模型完整指南.md`。
>
> 第四节是**可照跑的完整基准**（Colab / Kaggle + Qwen2.5-7B）：安装格、基准代码、口径、降档预案、结果表**全在本文档内**，没有单独的脚本文件。
>
> **一句话结论**：Unsloth = HF 生态之上的**算子级加速层**——接口兼容 HF，内核替换 HF。

---

## 一、先分清两种对照

| | 代码对照 | 性能对照 |
|---|---|---|
| 比什么 | 两侧代码结构的差异 | 同一任务下的耗时与显存 |
| 前提 | 无（模型、数据可以不同） | **必须同模型、同数据、同超参** |
| 在哪节 | 第二节 | 第四节 |

代码对照看的是"Unsloth 把哪几行吞掉了"，与模型大小无关；性能对照要的是数字，条件不统一就没有意义。

---

## 二、代码对照

### 2.1 原生 HF 侧

基准脚本：`llm-lora-qlora-finetuning-guide/finetune_llm_with_lora_and_qlora.py`（实际产出 `opt-125m-lora-adapter` 与 `opt-125m-merged` 的那个）。

**LoRA 段**（第 22–31 行）：

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))
```

**QLoRA 段**（第 52–65 行）——量化配置是内联的，且加载后有两行手工处理：

```python
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
```

其中第 59–60 行这两行是关键：`use_cache = False` 规避 KV 缓存与梯度检查点的冲突，`enable_input_require_grads()` 让梯度能穿过量化层到达 LoRA adapter。（`qlora_finetune_opt.py` 第 31–32 行给这两行加了同样的注释。）

### 2.2 Unsloth 侧

基准脚本：`unsloth-finetuning-guide/finetune_basic.py`（与 `Unsloth微调实战指南.md` 4.1 代码块逐字一致）。

```python
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
```

### 2.3 对照表：谁吞掉了什么

| 环节 | 原生 HF | Unsloth | 净效果 |
|---|---|---|---|
| 量化配置 | `BitsAndBytesConfig(load_in_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, bnb_4bit_use_double_quant)` 四个参数 | 收进 `from_pretrained(load_in_4bit=True)` | **4 行 → 0 行** |
| 模型加载 | `AutoModelForCausalLM.from_pretrained(quantization_config=…, device_map="auto")` | `FastLanguageModel.from_pretrained(...)` | 6 行 → 1 行 |
| k-bit 准备 | 手写 `model.config.use_cache = False` + `model.enable_input_require_grads()` | 无（`get_peft_model` 内部处理） | **2 行 → 0 行** |
| LoRA 配置 | `LoraConfig(...)` + `get_peft_model(...)` 两步 | `FastLanguageModel.get_peft_model(...)` 一步 | 合并为 1 个调用 |
| 训练器 | `SFTTrainer` + `SFTConfig` | `SFTTrainer` + `SFTConfig` | **完全没变** |
| 保存 | `save_pretrained` / `merge_and_unload` | 同左，另有 `save_pretrained_gguf`（直接出 GGUF） | 多一项导出能力 |

### 2.4 两个关键发现

**① 原生侧从来没用过 `prepare_model_for_kbit_training`。** 全目录搜索 0 匹配——实际是用 `use_cache = False` + `enable_input_require_grads()` 两行手写替代的。而 Unsloth 把这段整个收进了一个 `get_peft_model`。**这是两侧差异最集中的地方。**

**② `SFTTrainer` 部分两边一模一样。** 这说明此前踩的那批报错（`evaluation_strategy` → `eval_strategy`、`warmup_ratio` → `warmup_steps`、`tokenizer` → `processing_class`）**与 Unsloth 没有关系**，那是 TRL 1.12 的 API 更名——原生、Unsloth 两侧都得跟着改。

> **附注**：两侧的数据模板不同——原生侧脚本用 `### Human / ### Assistant`（见 `lora_finetune_opt.py` 第 11–12 行），Unsloth 侧 `finetune_basic.py` 用 Alpaca 的 `### Instruction / ### Response`。**这是数据格式的选择差异，不是 Unsloth 带来的差异。**

---

## 三、Unsloth 为什么快

它不是"包一层"——**包一层只会更慢**。速度来自底层实现被换掉了：

| 算子 | HF / PyTorch 原生 | Unsloth |
|---|---|---|
| 注意力（attention） | 通用 kernel | 手写 Triton kernel |
| MLP（前馈层） | 通用 kernel | 手写 Triton kernel |
| 交叉熵损失 | 通用 kernel | 手写 Triton kernel + 手动反向传播 |
| RoPE / RMSNorm | 通用 kernel | 手写 Triton kernel |

关键点：**数学等价，不是近似**。速度与显存收益来自"手动推导反向传播 + 复用中间结果（不把全部激活存下来）"，而非牺牲精度。

HuggingFace 官方博客的原话：

> "Unsloth works by **overwriting some parts of the modeling code** with optimized operations. By **manually deriving backpropagation steps** and **rewriting all PyTorch modules into Triton kernels**, Unsloth can both reduce memory usage and make fine-tuning faster."
>
> （Unsloth 的做法是**用自己的优化实现覆盖部分建模代码**：手动推导反向传播，并把 PyTorch 模块改写成 Triton 内核，因此同时降低显存占用、加快微调速度。）
>
> 来源：https://huggingface.co/blog/unsloth-trl

---

## 四、性能对照：怎么测

### 4.1 前提：必须同条件

条件不对齐，数字就没有意义。两侧必须完全一致——下表就是 4.3 那段代码顶部「公共口径」区的全部内容：

| 项 | 值 |
|---|---|
| 基座 | `Qwen/Qwen2.5-7B`（两侧同一份权重，各自加载时做 4-bit nf4 量化） |
| 数据 | `yahma/alpaca-cleaned`，`shuffle(seed=42).select(range(2000))` |
| 模板 | Alpaca `### Instruction / ### Input / ### Response` |
| 最大长度 | 2048 |
| 步数 | `max_steps=60`（**不用 epochs**——否则两侧数据量不同就不可比） |
| batch / 累积 | 2 / 4 |
| 学习率 | 2e-4，cosine，warmup 10 步 |
| 优化器 | `adamw_8bit` |
| LoRA | r=16 / alpha=32 / dropout=0 / `target_modules` 7 个（q,k,v,o,gate,up,down） |
| **可训练参数量** | **40,370,176**（两侧必须一模一样——这是"同口径"唯一的硬指标，代码在构造 trainer 后断言） |
| 精度 | fp16（T4 无 bf16） |
| eval | 两侧都关闭 |

几条容易踩的：

1. **不用 `unsloth/Qwen2.5-7B-bnb-4bit`。** 预量化权重和加载时量化可能不完全等价，会被质疑"权重来源不一致"。两侧都用官方 `Qwen/Qwen2.5-7B`。
2. **`target_modules` 必须都是 7 个。** 仓库里 `finetune_basic.py` 是 7 个、原生脚本是 4 个——直接拿那两个脚本比，可训练参数量不同，耗时不可比。
3. **精度写死 fp16。** 别用 `torch.cuda.is_bf16_supported()` 自动判断，换到 L4 / A100 两侧就会不一致。
4. **原生侧不要手动装 Flash Attention 2**，用 PyTorch 默认的 SDPA 即可（T4 上装 FA2 麻烦，且会改变对比口径）。
5. **梯度检查点两侧对称**：Unsloth 用 `use_gradient_checkpointing="unsloth"`，原生用 `gradient_checkpointing_enable()`（写在 `load_model()` 的 native 分支里）。这是被测量的差异本身，不算不公。**别图省事把它挪进共用的 `SFTConfig`**：那会让 TRL 调标准的 `gradient_checkpointing_enable()` 去覆盖 Unsloth 自己的 GC，等于动了被测对象。
6. **显存口径**：`reset_peak_memory_stats()` 在 `train()` **之前**调用，所以报的是**训练阶段峰值，不含模型加载**。填表时别当成整机峰值。
7. **可训练参数量是"同口径"的硬指标**：4.3 代码在**构造 trainer 之后**断言一次，必须等于 **40,370,176**——不等就不跑，省得白等一场训练。位置之所以在 trainer 之后：原生侧的 adapter 是 `SFTTrainer` 在构造过程中挂的。原生侧实测踩过一次 `trainable params: 0`（adapter 挂上了却被冻住），见 **4.8 ④**。

### 4.2 环境准备：安装格（两个 notebook 必须用逐字相同的这一格）

取自 Unsloth 官方 notebook 的标准安装格，**去掉了 if/else 分支**，Colab / Kaggle 通用：

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

几条必须知道的：

- **不要加 `%%capture`**：它会把 pip 的报错一起吞掉，装失败时屏幕上看不出任何异常。多几十行日志换"报错可见"，划算。
- **这几行顺序是有意的**：先 `--no-deps` 装 trl，再单独把版本钉到 0.22.2，防止 trl 的依赖把 transformers 拽走。别重排。
- **不升级 torch**（只读版本号来选 xformers）：在**全新会话里"先装后跑"，不需要重启**。但如果这个会话已经 `import` 过 transformers / trl（比如代码已经跑过一次），装完**必须重启会话**——否则内存里还是旧版本，代码读到的仍是 5.x。
- **xformers 的版本映射只覆盖 torch 2.8 / 2.9 / 2.10**，其它版本会 fallback 到 `0.0.34`。若日后 `import xformers` 报 ABI 错，把 `{xformers}` 从安装格里去掉即可——**两侧都要去**。
- **版本被钉在** `transformers==4.56.2` / `trl==0.22.2`。4.3 代码用到的 `processing_class=` / `max_length=` / `eval_strategy=` 在这个组合里都有，不会因版本报错。但**别在同一个会话里跑项目里按 transformers 5.x 写的其他脚本**。
- **xformers 会同时装进两侧**（条件因此仍一致），**绝不能只在一侧装**。
- **为什么去掉 if/else**：官方格用 `"COLAB_" not in os.environ` 判断平台——本地走 `!pip install unsloth`，云端才走上面这几行。但这个基准只在云端跑，那条本地分支永远执行不到；更麻烦的是**一旦 `COLAB_` 没被检测到，整段安装会被静默跳过、还不报错**（这种失败极难排查）。写成无条件版，少一个失败模式。

> **可选提速**：这一格已经装了 `hf_transfer`。在 4.3 那段代码**最开头**加一行 `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"`，下 15GB 的 `Qwen/Qwen2.5-7B` 会快不少。**两侧都要加。**

**装错了怎么认**

| 现象 | 说明 |
|---|---|
| `ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes` | 这个 notebook 没跑安装格，或跑了但失败了 |
| 版本表里 `trl : (未安装)` / `bitsandbytes : (未安装)` | 同上——这两个包都是安装格装的，Colab 不预装 |
| `transformers : 5.17.0`（不是 4.56.2） | 安装格没生效 |
| `ModuleNotFoundError: No module named 'structlog'`（或 tyro / msgspec / cut_cross_entropy） | 安装格跑成功了，但漏了 `--no-deps` 带出来的运行期依赖。补 `!pip install --no-deps structlog tyro msgspec cut_cross_entropy`，重启会话。**只在 Unsloth 侧会遇到** |
| `You are sending unauthenticated requests to the HF Hub` | 只是提示，`Qwen/Qwen2.5-7B` 不需要 token；被限流时再设 `HF_TOKEN` |

**根因只有两个**：① Colab 的运行时**按 notebook 独立**——在别的 notebook 装过不算，每个 notebook 都要单独跑一次这一格；② 安装中途失败了你却没看见（习惯给安装格加 `%%capture` 的人常踩这个，所以本项目的安装格刻意不加）。

**pip 结尾那堆冲突警告，哪些要管**——正常装完会打印一长串 `ERROR: pip's dependency resolver ...`。逐条对照，**大部分是噪音**：

| 警告原文 | 要不要管 |
|---|---|
| `unsloth ... requires structlog / tyro, which is not installed`<br>`unsloth-zoo ... requires msgspec / cut_cross_entropy / tyro, which is not installed` | **要管**。安装格对这两个包用了 `--no-deps`，它们的运行期依赖不会被自动装。若 `import unsloth` 报错，补一行 `!pip install --no-deps structlog tyro msgspec cut_cross_entropy` 再重启 |
| `unsloth ... requires trl<=0.24.0, but you have trl 1.14.1`<br>`trl 1.14.1 requires datasets>=4.7.0, but you have datasets 4.3.0` | **不管**。"钉版本"之前的中间态：Colab 预装的 trl 1.14.1 这时还没被换掉，最后一行 `!pip install --no-deps trl==0.22.2` 跑完就消失 |
| `gradio ... requires huggingface-hub>=1.16.0, but you have huggingface-hub 0.36.2`<br>`diffusers ... requires huggingface-hub>=1.23.0` | **不管**。Colab 预装的，跑微调用不到；而 `transformers==4.56.2` 本身就要求 `huggingface-hub<1.0`，**降到 0.36.2 是预期行为** |

**装成功的判据**——最后两行必须长这样（顺序可能不同）：

```
Successfully installed huggingface-hub-0.36.2 tokenizers-0.22.2 transformers-4.56.2
Successfully installed trl-0.22.2
```

下面 4.3 那段代码里已经加了预检：环境不对会立刻停下并告诉你缺什么；**Unsloth 侧还会先真 `import unsloth` 试一次**，所以那四个漏装的依赖会在**下 15GB 权重之前**就被拦下，不会白等。

### 4.3 基准代码（两侧共用，唯一真源）

**代码全仓库只有这一份**——粘进 Colab 那个 cell 的就是它本身，没有独立的 `.py`，不会出现"文件改了、文档没跟上"。跑之前只改下面这一行：

```python
SIDE = "unsloth"   # 另一个 notebook 填 "native"，其余一字不改
```

为什么只有一份：同条件对照最大的风险是"两侧配置悄悄漂移"。数据、超参、采集代码物理上共用同一段，只有 `load_model()` 分叉——口径不可能不一致。

```python
"""Qwen2.5-7B QLoRA 同条件性能基准：Unsloth vs 原生 HuggingFace

用途：跑出《Unsloth与原生HuggingFace对照.md》第四节 4.6 结果表要填的数字。

怎么跑（Colab / Kaggle 免费 GPU，T4 16GB）：
    1) 先建一个 notebook，跑 4.2 的"环境准备格"
    2) 把下面这整段粘进下一个 cell
    3) 只改下面 SIDE 这一行，跑一次
    4) 再新建一个 notebook（或重启运行时）跑另一侧

⚠️ 两侧必须在两个独立 notebook / 独立进程里跑。
   import unsloth 会在导入时全局改写 HF 的模型实现，
   同一个会话里连跑两侧，原生侧会被污染，数字作废。
"""

SIDE = "unsloth"  # ← 只改这一行："unsloth" 或 "native"

import torch

# ── 公共口径：两侧必须完全一致 ──────────────────────────────────
# 这两个分支只允许在 load_model() 里分叉，禁止在分支内覆盖下面的任何常量。
MODEL_NAME = "Qwen/Qwen2.5-7B"      # 两侧同一份权重，各自在加载时做 4-bit 量化
DATASET_NAME = "yahma/alpaca-cleaned"
NUM_SAMPLES = 2000                  # 固定抽样条数（shuffle + select，种子固定）
SEED = 42

MAX_SEQ_LEN = 2048
MAX_STEPS = 60                      # 固定步数（不用 epochs），保证两侧训练量相同
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
LR_SCHEDULER = "cosine"
OPTIM = "adamw_8bit"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
# 两侧必须同为 7 个（含 MLP），否则可训练参数量不同，耗时不可比
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

OUTPUT_DIR = f"./bench-{SIDE}"

# T4 不支持 bf16 → 两侧都强制 fp16。
# 不要用 torch.cuda.is_bf16_supported() 自动判断：换到 L4 / A100 会让两侧口径不一致。
COMPUTE_DTYPE = torch.float16

# Unsloth 侧实测的可训练参数量（r=16 + 上面 7 个 target_modules 的 LoRA 参数量）。
# 两侧必须完全相同：不等就说明 target_modules / LORA_R / LORA_ALPHA 有一侧漂了，数字作废。
EXPECTED_TRAINABLE = 40_370_176


def trainable_param_count(model):
    """只数 requires_grad=True 的参数（与 PEFT 的 print_trainable_parameters 同一口径）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_prompt(sample):
    """Alpaca 模板（与 unsloth-finetuning-guide/finetune_basic.py 一致）"""
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


def build_dataset():
    """固定种子抽样，两侧得到逐字相同的数据"""
    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, split="train")
    ds = ds.shuffle(seed=SEED).select(range(NUM_SAMPLES))
    return ds.map(lambda x: {"text": format_prompt(x)})


def load_model():
    """唯一允许两侧分叉的地方。返回 (model, tokenizer, peft_config)。

    peft_config 是两侧唯一的"形态差异"：
      Unsloth 侧 = None            —— adapter 已由 FastLanguageModel.get_peft_model 挂好
      原生侧   = 一份 LoraConfig    —— 交给 SFTTrainer 按官方顺序去挂（为什么必须这样见 4.8 ④）
    """
    if SIDE == "unsloth":
        from unsloth import FastLanguageModel  # 延迟导入：只有这一侧才 import
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            dtype=COMPUTE_DTYPE,        # 必须显式传：Qwen2.5-7B 的 config 是 bfloat16，T4 不支持
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )
        # adapter 已经挂好了 → peft_config 传 None，让 SFTTrainer 原样使用这个模型。
        return model, tokenizer, None

    if SIDE == "native":
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, TaskType

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=COMPUTE_DTYPE,   # 同上：config 声明 bfloat16，必须显式覆盖成 fp16
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=COMPUTE_DTYPE,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
        )
        model.config.use_cache = False          # 与梯度检查点不兼容
        model.gradient_checkpointing_enable()   # 对齐 Unsloth 侧的 use_gradient_checkpointing
        model.enable_input_require_grads()      # 让梯度穿过冻结的基座，抵达 LoRA adapter

        # ── 这里【不】挂 adapter，只把 LoraConfig 交给 SFTTrainer ──────────
        # 原生侧必须让 SFTTrainer 自己挂，因为 TRL 内部的顺序是固定的：
        #     prepare_model_for_kbit_training()   ← 把【所有】参数冻住
        #   → get_peft_model()                    ← 再把 adapter 放开
        # 我们要是自己先 get_peft_model，TRL 的第二步"全冻"就会把 adapter 一起冻死，
        # 可训练参数量变成 0，训练以 "No inf checks were recorded" 收场。详见 4.8 ④。
        #
        # 梯度检查点为什么写在这里、而不是写进共用的 SFTConfig：
        #   SFTConfig(gradient_checkpointing=True) 会让 TRL 去调标准的
        #   gradient_checkpointing_enable()，那头会把 Unsloth 自己的 GC 覆盖掉，
        #   等于改了被测量对象。原生侧在这里手动开，Unsloth 侧维持 "unsloth" 那一套。
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=TARGET_MODULES,
        )
        return model, tokenizer, peft_config

    raise ValueError(f'SIDE 只能填 "unsloth" 或 "native"，当前是 {SIDE!r}')


def build_trainer(model, tokenizer, train_dataset, peft_config):
    """两侧共用同一个训练器配置。

    peft_config 不是"超参"，它由 load_model() 决定（两侧形态不同，见那里的注释），
    所以不违反"只有 load_model() 可以分叉"这条规矩：
    LORA_R / LORA_ALPHA / LORA_DROPOUT / TARGET_MODULES 依旧只有一处取值。
    """
    from trl import SFTConfig, SFTTrainer
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            max_length=MAX_SEQ_LEN,
            max_steps=MAX_STEPS,            # 固定步数，不用 num_train_epochs
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            lr_scheduler_type=LR_SCHEDULER,
            optim=OPTIM,
            fp16=True,                      # T4 无 bf16
            bf16=False,
            eval_strategy="no",             # 两侧都关 eval
            save_strategy="no",             # 不写 checkpoint，避免磁盘干扰
            logging_steps=10,
            report_to="none",
            seed=SEED,
        ),
    )


# ── 主流程 ────────────────────────────────────────────────────
import importlib.metadata as md

print("=" * 60)
print("对照侧        :", SIDE)
print("模型          :", MODEL_NAME)
print("设备          :", torch.cuda.get_device_name(0))
print("显存(GB)      :", torch.cuda.get_device_properties(0).total_memory / 1e9)
for pkg in ("torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate"):
    try:
        print(f"{pkg:<14} :", md.version(pkg))
    except md.PackageNotFoundError:
        print(f"{pkg:<14} : (未安装)")
print("=" * 60)

# ── 预检：确认"环境准备格"在本 notebook 里真的跑成功了 ────────────────
# 少装一个包，报错会发生在很深的地方（例如 bitsandbytes 要到加载模型时才炸），
# 这里提前拦下来，直接告诉你去做什么。
import importlib.util

_missing = [p for p in ("trl", "bitsandbytes", "peft", "accelerate", "datasets")
            if importlib.util.find_spec(p) is None]
if _missing:
    raise RuntimeError(
        "缺少依赖：" + ", ".join(_missing) + "\n"
        "→ 这个 notebook 没跑过（或没跑成）4.2 的环境准备格。检查三件事：\n"
        "  1) Colab 的运行时按 notebook 独立，每个 notebook 都要单独跑一次安装格；\n"
        "  2) 安装格里的 %%capture 会吞掉报错，调试时先删掉它；\n"
        "  3) 若本会话已 import 过 transformers，装完要重启会话。"
    )

# ⚠️ 这里刻意【不】import transformers。
#   Unsloth 必须在 transformers 之前导入，否则它的一部分优化不生效，会打出
#   "Unsloth should be imported before [transformers]" 警告 —— 那对照就不公平了。
#   查版本用 importlib.metadata 就够了，它只是读元数据，不会把包导进内存。
#   因此本项目全程不出现 `import transformers`，下面那次 unsloth 导入是第一次导入。
_tf_ver = md.version("transformers")
if _tf_ver.split(".")[0] != "4":
    print("⚠️ transformers =", _tf_ver,
          "；官方安装格会把它钉成 4.56.2 —— 确认安装格真的跑过了。")

# 安装格对 unsloth / unsloth_zoo 用的是 --no-deps，它们自己的运行期依赖
# （structlog、tyro、msgspec、cut_cross_entropy）不会被自动装上。
# 缺了会在 import 这一层就炸 —— 那就放在这里炸，还顺手带上修复命令，
# 而不是等 15GB 权重下完再报。
if SIDE == "unsloth":
    try:
        from unsloth import FastLanguageModel  # noqa: F401
    except ModuleNotFoundError as _e:
        raise RuntimeError(
            f"import unsloth 失败：缺少 {_e.name}\n"
            "→ 安装格用了 --no-deps，unsloth / unsloth_zoo 的运行期依赖不会自动装。补一次：\n"
            "    !pip install --no-deps structlog tyro msgspec cut_cross_entropy\n"
            "  装完重启会话，再把上面整段代码重新粘一遍。\n"
            "  （务必带 --no-deps：不带会把 transformers / torch 拽走。）"
        ) from _e

# ── 加载前清场：必须放在 load_model() 之前 ─────────────────────────
# 同一个运行时里重跑这段代码（只重新运行 cell、没有真正重启）时，上一次的
# model / trainer 仍然活着、仍占着显存，新的加载就会在"分片还剩一半"时 OOM。
# 症状极具迷惑性：报错看着像"显存不够"，其实是上一次的残留。
# 这几个名字都要清：trainer 持有 model 的引用，只 del model 是清不掉的。
import gc

for _name in ("model", "trainer", "train_dataset", "tokenizer", "peft_config"):
    if _name in globals():
        del globals()[_name]
gc.collect()
torch.cuda.empty_cache()

_free, _total = torch.cuda.mem_get_info()
print("加载前空闲显存(GB): %.2f / %.2f" % (_free / 1e9, _total / 1e9))
if _free / 1e9 < 12:
    raise RuntimeError(
        "加载前 GPU 空闲显存只有 %.1f GB（需要 ≥12GB）。\n"
        "→ 显存被上一次运行或其它进程占着，别硬跑，先清干净：\n"
        "  Colab：运行时 → 断开连接并删除运行时（Disconnect and delete runtime）；\n"
        "  Kaggle：Run → Restart & Clear Cell Outputs（或 Factory reset）。\n"
        "  注意：只重新运行 cell 不算重启，上一次的 model 还活着。" % (_free / 1e9)
    )

model, tokenizer, peft_config = load_model()

# ── 4-bit 自检：量化没生效就当场停下，别等到 OOM ────────────────────
# 7B 正确做了 4-bit 只占 ~4.4GB。若这里报出十几 GB，说明量化被静默绕过，
# 权重是按 fp16 全量加载的 —— 继续跑必然 OOM。
_alloc = torch.cuda.memory_allocated() / 1e9
print("加载后显存(GB): %.2f" % _alloc)
# 诊断用：TRL 就是按下面这两个事实，决定"要不要给这个模型再补一次 4-bit 预处理"的
# —— 而那一步会把所有参数冻住（见 4.8 ④）。两侧日志对比一下，这条坑为什么只在
# 原生侧发作就有答案了，不必猜。
_first4 = next((p for p in model.parameters() if p.__class__.__name__ == "Params4bit"), None)
print("4-bit 判定    :", getattr(model, "is_loaded_in_4bit", "（无此属性）"),
      "| 首个 4bit 参数在", "（无）" if _first4 is None else _first4.data.device.type)
if _alloc > 8:
    raise RuntimeError(
        "4-bit 量化似乎没有生效：模型加载后已占 %.1f GB（正确应 ~4.4GB）。\n"
        "→ 大概率是 dtype 不兼容：Qwen2.5-7B 的 config 声明 bfloat16，而 T4 不支持。\n"
        "  确认 from_pretrained 里显式传了 dtype=COMPUTE_DTYPE（Unsloth 侧）/ \n"
        "  torch_dtype=COMPUTE_DTYPE（原生侧），且 COMPUTE_DTYPE 是 torch.float16。\n"
        "  另外两侧都别再手动传 bf16 相关参数。" % _alloc
    )

train_dataset = build_dataset()
trainer = build_trainer(model, tokenizer, train_dataset, peft_config)

# ── 梯度自检：可训练参数量必须与 Unsloth 侧完全相同 ─────────────────
# 位置很关键：必须在 build_trainer() 之后，而且数的必须是 trainer.model。
# 原生侧的 adapter 是 SFTTrainer 在构造过程中挂的，构造之前根本没有 adapter 可数。
# 这是本基准唯一的"同口径"硬指标——两侧的 LoRA 必须是同一批、同样的数量。
# 原生侧曾出现 "trainable params: 0"（adapter 挂上了却被冻住），症状是训练时抛
# "No inf checks were recorded for this optimizer"——报错点离真因很远，所以提前拦下。
_trainable = trainable_param_count(trainer.model)
print("可训练参数量  : %s / 期望 %s" % (f"{_trainable:,}", f"{EXPECTED_TRAINABLE:,}"))
if _trainable != EXPECTED_TRAINABLE:
    raise RuntimeError(
        "可训练参数量 = %s，期望 %s（Unsloth 侧实测值）。\n"
        "→ 0 或明显偏小：adapter 没挂上 / 被冻住（见 4.8 ④）；\n"
        "→ 数量接近但不等：target_modules、LORA_R、LORA_ALPHA 有一侧漂了。\n"
        "  两种情况都会让耗时不可比，先修再跑。" % (f"{_trainable:,}", f"{EXPECTED_TRAINABLE:,}")
    )

trainer.model.print_trainable_parameters()
print("-" * 60)

# ↓↓↓ 采集段：四项指标即 4.6 结果表的来源，改动要同步 4.1 的口径表 ↓↓↓
import torch
torch.cuda.reset_peak_memory_stats()          # 必须在 train 之前调用

res = trainer.train()

print("耗时(s)      :", res.metrics["train_runtime"])
print("每秒样本数   :", res.metrics.get("train_samples_per_second"))
print("峰值显存(GB) :", torch.cuda.max_memory_allocated() / 1e9)
print("保留显存(GB) :", torch.cuda.max_memory_reserved() / 1e9)
# ↑↑↑ 采集段 ↑↑↑
```

跑完把控制台输出的那段（设备 / 版本 / 可训练参数 / 四项指标）抄进 **4.6 的表**。

> Colab 免费档给的是 **T4 16GB**。Qwen2.5-7B 的 4-bit QLoRA 在 T4 上两侧都能跑。遇到 OOM 先看 **4.8 ①** 分清是"运行时没清干净"还是"量化没生效"，**别直接降档**——两者都不是显存不够。

### 4.4 两侧怎么跑：四条硬规则

在 Colab / Kaggle 各建**两个独立 notebook**，一次只跑一侧（先跑哪侧不限）：

1. **两侧必须独立 notebook / 独立进程。** `import unsloth` 会在导入时全局改写 HF 的模型实现，同一会话里连跑两侧，原生侧的数字直接作废。
2. **每开一侧，先真正重启运行时。** 「重新运行 cell」不算重启：上一次的 `model` / `trainer` 还在显存里，新一次加载会在分片 50% 处 OOM（见 4.8 ①）。Colab「断开连接并删除运行时」/ Kaggle「Restart & Clear Cell Outputs」。4.3 代码里已加了自动清场 + 空闲显存断言，会帮你兜底，但别依赖它。
3. **两侧必须同一型号 GPU。** 4.3 代码第一段就会打印 `torch.cuda.get_device_name(0)`；T4 与 L4 的数字不可混在一张表里。Colab 分配是随机的，两次跑之前都确认一下。
4. **跑 2~3 次取平均。** 单次数字有噪声，尤其耗时。

另外，两侧打印的库版本（torch / transformers / trl / peft / bitsandbytes）也应一致——这就是为什么要用 4.2 那一格逐字相同的安装格。

> **导入顺序也算一条口径（Unsloth 侧）**：必须让 Unsloth 比 `transformers` 先导入。若先 `import transformers`，Unsloth 会打出
>
> ```
> UserWarning: WARNING: Unsloth should be imported before [transformers] to ensure all
> optimizations are applied. Your code may run slower or encounter memory issues...
> ```
>
> 也就是说**它的部分优化没生效**——那样测出来的耗时对 Unsloth 不公平。4.3 的代码已经规避了这点：全篇不出现 `import transformers`，查版本改用 `importlib.metadata`（只读元数据、不导入包），所以预检里那次 `from unsloth import FastLanguageModel` 就是第一次导入。**看到这条警告，说明数字要重跑。**
>
> 本项目实测踩过一次：有警告那次 416.46 s；修掉后三次干净运行 370.80 / 343.30 / 350.82 s（均值 **354.97 s**），差距约 **17.3%**（对照见 4.6 的运行记录）。方向一致，但 n=1 对 n=3 仍不宜当定论——**至少别拿单次跑出来的差值下结论**。

### 4.5 降档预案

只有一条铁律：**要降就两侧一起降，绝不能只降一边。**

| 症状 | 处理 |
|---|---|
| 崩在分片 50%（已占≈全部显存） | **不是显存不够，是运行时没清干净** → 重启运行时后重跑，别动参数（见 4.8 ①） |
| 真·OOM：原生侧崩、Unsloth 侧正常 | 两侧同时 `MAX_SEQ_LEN` → 1024、`BATCH_SIZE` → 1、`GRAD_ACCUM` → 8 |
| 还 OOM | 两侧同时 `MAX_SEQ_LEN=512` |
| 7B 两侧都跑不动 | 两侧一起换 `Qwen/Qwen2.5-3B`，并在表下注明换了模型 |
| Colab 会话中断 | `MAX_STEPS` 调小（如 30）后两侧重跑；两侧步数必须一致 |

### 4.6 结果表

| 指标 | Unsloth | 原生 HF | 倍数 / 差异 |
|---|---|---|---|
| 训练耗时（s） | 354.97 † | | |
| 峰值显存（GB） | 8.76 † | | |
| 每秒样本数 | 1.353 † | | |
| 可训练参数量 | 40,370,176 † | | |

> † **Unsloth 侧 = 3 次干净运行的算术平均**（口径要求 2~3 次，已满足）。原生 HF 侧还没跑出来，所以"倍数 / 差异"栏全部留空——**两侧都齐了才算数**。

**Unsloth 侧运行记录**（原始数据，便于回溯）

| # | 状态 | 耗时（s） | 每秒样本数 | 峰值显存（GB） |
|---|---|---|---|---|
| 1 | ✗ 有导入顺序警告 + 运行时未重启 → 作废 | 416.46 | 1.153 | 8.76 |
| 2 | ✓ 干净 | 370.80 | 1.294 | 8.76 |
| 3 | ✓ 干净 | 343.30 | 1.398 | 8.76 |
| 4 | ✓ 干净 | 350.82 | 1.368 | 8.76 |
| — | **均值（#2 / #3 / #4）** | **354.97** | **1.353** | **8.76** |

> 三点读数提示：① 三次干净运行之间最大相差 **8.0%**（370.80 vs 343.30，均值 354.97），这是单次数字的噪声量级——所以"跑 2~3 次取平均"不是形式主义。② `每秒样本数` 取三次实测值的算术平均（1.294、1.398、1.368 → 1.353）；若用 `480 ÷ 354.97` 反推得 **1.352**，两者差 0.07%，属"平均速率 vs 平均耗时的倒数"的换算差异。③ 补第 3 次后，耗时均值比只用 #2 / #3 时微降 **0.6%**（现值 354.97 s），而单次极差仍有 8.0%——**均值稳定得多，这正是取平均的意义**。

**#1 为什么作废**：那一次的日志头部有 `Unsloth should be imported before [transformers]` 警告（详见 4.4 末尾），且运行时未重启（4.8 ①）。

它 **416.46 s**，比三次干净运行的均值 **354.97 s** 慢 **17.3%**；而三次干净运行彼此最大只差 **8.0%**——**这一档差距超出了噪声范围**，方向与"导入顺序影响优化是否全开"一致。但 **n=1 对 n=3 仍不足以定论**：带警告的运行只出现过一次，无法判断这 17.3% 里有多少是当次实例自身的波动。要下结论，得再复现一次"带警告运行"做对照。可以确定的是：那次过程不干净、教训成立，所以作废。

**四次运行完全相同的两项**：峰值显存 **8.76 GB**、六步 loss 曲线（1.154400 / 1.097500 / 1.026000 / 1.063100 / 0.976100 / 1.122600）**逐位相同**——**包括那次带警告的 #1**。也就是说**差别只落在速度上，数学与显存都没变**：与第三节"Unsloth 的加速来自等价的内核替换"一致，也是"快 17% 不是因为少算了什么"的最直接证据。

> **附带证据（新增，2026-10-03）**：第 4 次干净运行是在**4.3 代码加了"可训练参数量"自检之后**跑的，日志里出现 `可训练参数量  : 40,370,176 / 期望 40,370,176`——**加了 4-bit 自检与可训练参数量自检（含一次全参数遍历），耗时仍是 350.82 s（落在三次均值的噪声带内）**。也就是说：那些自检开销小到测不出来，不会污染耗时口径。

> **填表前先自检**：两侧"可训练参数量"必须都是 **40,370,176**。不一致说明 `target_modules` / `r` / `alpha` 有一侧漂了，这组数字作废。（4.3 代码里已断言两次，正常情况下根本跑不到填表这一步。）

**环境脚注（每次都要填）**

- GPU 型号：**Tesla T4**（`torch.cuda.get_device_name(0)`；报告 15.64 GB / 可用 14.56 GiB）
- 库版本（Unsloth 侧实测，**原生侧必须逐项一致**）：torch **2.11.0+cu130** / transformers **4.56.2** / trl **0.22.2** / peft **0.21.0** / bitsandbytes **0.50.2**
- `max_steps=60`，`max_length=2048`，batch 2 × accum 4
- `target_modules`：q,k,v,o,gate,up,down（7 个）
- 跑了几次、是否取平均：Unsloth 侧 **3 次干净运行取算术平均（已满足口径的 2~3 次）**；原生侧待跑
- 加载后显存：Unsloth 侧 **7.73 GB**（自检阈值 <8 GB 通过，说明 4-bit 生效）
- **原生侧进展**：加载正常（加载后 **5.72 GB**，4-bit 同样生效）。先前卡在 `trainable params: 0`，真因已定位并修好（见 **4.8 ④**，2026-10-03 改的代码）——**代码改后重跑一次，才有原生列数字**。
- **Unsloth 列的 3 次数据不受这次改动影响**：改动只动了 native 分支和"在哪里数可训练参数量"，Unsloth 侧仍传 `peft_config=None`、仍由 `FastLanguageModel.get_peft_model` 挂 adapter，**无需重跑**。
- 「可训练参数量」两侧应完全相同——不一致说明口径漂了，数字作废

### 4.7 官方公开数字（**非本人实测，仅作量级参考**）

> 来源：HuggingFace 官方博客 https://huggingface.co/blog/unsloth-trl
> 对比基线：**"HF + Flash Attention 2"**；测试环境：A100 40GB 与免费 Colab T4。

| 环境 | 模型 / 数据 | 加速 | 显存降幅 |
|---|---|---|---|
| A100 40GB | Llama-2 7b / Slim Orca | 1.87× | -39.3% |
| A100 40GB | Mistral 7b / Slim Orca | 1.88× | -65.9% |
| A100 40GB | Tiny Llama 1.1b / Alpaca | 2.74× | -57.8% |
| **免费 Colab T4** | Llama-2 7b / OASST | **1.95×** | **-43.3%** |
| 免费 Colab T4 | Mistral 7b / Alpaca | 1.56× | -13.7% |

**版本注意**：这批数据来自 transformers 4.36 / PyTorch 2.1.1 时期，与当前版本（TRL 1.12 / transformers 5.16）不同，仅作量级参考。要形成自己的结论，请以 4.6 的实测数字为准。

### 4.8 已知限制与已踩的坑

**① 7B 在 T4 上 OOM，先分清是哪一类，别一上来就降档。**

实测踩过三次，症状不同、根因**两类**——不要把它们混成一个问题：

| 现场 | 崩在哪 | 关键数字 | 根因 |
|---|---|---|---|
| 原生 HF | `SFTTrainer(...)` → `prepare_model_for_kbit_training` 里 `param.data.to(torch.float32)` | 崩前已占 12.71 GiB，失败分配 2.03 GiB（= 545M 参数实体 fp32） | **量化没生效**：`from_pretrained` 没显式传 dtype，吃了 config 的 bf16 |
| Unsloth（第 1 次） | `load_model()` → loading checkpoint shards **50%** | 已占 14.27 GiB | **运行时没清场**：上一次的 model 还活着 |
| Unsloth（第 2 次，同一份代码） | 同上 | 同上（数字逐字节相同） | 同上 |
| Unsloth（清干净后） | 加载成功 | 加载后 **7.73 GB** | 正常 |

**两条判据，一眼就能分开：**

1. **崩在 50%** —— 若"已占显存 ≈ 全部显存"而"进度只有一半"，这不可能是一次干净加载（7B 加载到一半只需约 7GB）。差额约 7.2GB 就是**上一次运行残留的模型**。**先清场，不要改任何参数。** 同一份代码在干净运行时能加载成功（7.73GB），就是最直接的证明。
2. **加载后 `memory_allocated() > 8GB`** —— 量化没生效（正确值 ~4.4GB；实测 7.73GB 是含加载开销的正常值）。4.3 的代码里已加这条自检，会当场抛错。

**残留这一类怎么修**：`load_model()` 之前必须清场（4.3 代码里已加，会自动 `del model / trainer / train_dataset / tokenizer` + `gc.collect()` + `empty_cache()`，再断言空闲 ≥12GB）。
**最容易犯的错是"只重新运行 cell"当作重启**——`trainer` 持有 `model` 的引用，不清就一直在显存里。正确的重启：Colab「断开连接并删除运行时」/ Kaggle「Restart & Clear Cell Outputs」。**每一侧开跑前都重启一次**，开销只是重新跑一遍安装格。

> **已证实**：按上面的方式重启后再跑，日志出现 `加载前空闲显存(GB): 15.53 / 15.64`，紧接着加载成功（`加载后显存(GB): 7.73`）。**同一份代码、同一个 notebook，前两次崩、这次通**——差别就只在显存是不是干净的。这行"加载前空闲显存"平时应当接近 15.5GB；只有 8GB 上下就说明有残留。

**量化没生效那一类的根因**：`Qwen2.5-7B` 的 `config.torch_dtype` 是 **`bfloat16`**，而 **T4 不支持 bf16**（capability 7.5，`is_bfloat16_supported() = False`）。所以必须在 `from_pretrained` 里**显式**把 dtype 钉成 fp16——Unsloth 侧传 `dtype=`、原生侧传 `torch_dtype=`，两侧都不能省。

**排查步骤**（10 秒定位，别靠猜）：

```python
import torch, bitsandbytes as bnb
from transformers import AutoConfig

# 0) 崩了之后先看残留：数字很大（约 7GB）= 上一次的 model 还活着 → 重启运行时
print("残留       : %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

# 1) bnb 的 4-bit 内核在这张卡上能不能跑（应该输出 torch.uint8）
q, _ = bnb.functional.quantize_4bit(torch.randn(512, 512, device="cuda", dtype=torch.float16))
print("4-bit 内核:", q.dtype)

# 2) 模型实际占了多少 —— >8GB 就是量化没生效
print("allocated : %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

# 3) config 声明的 dtype（bf16 = 需要显式覆盖）
print("config    :", AutoConfig.from_pretrained("Qwen/Qwen2.5-7B").torch_dtype)
```

**② 这个 OOM 发生在"准备/加载"阶段，调 `BATCH_SIZE` / `MAX_SEQ_LEN` 完全无效。** 那些只影响训练时的激活值，模型权重该占多少还占多少。别去试。

**③ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**（报错信息自己会建议）必须在**本会话第一次 `import torch` 之前**设置才生效。而安装格（4.2）里就有 `import torch`——所以**放进 4.3 那段代码是没用的**，要加只能加到 4.2 安装格的最前面，或更早的一格。

**④ 原生侧 `trainable params: 0`——adapter 挂上了，却被冻住。**（**根因已定位，2026-10-03**）

现象：加载正常、数据正常，`print_trainable_parameters()` 却打印

```
trainable params: 0 || all params: 7,655,986,688 || trainable%: 0.0000
```

随后训练抛 `AssertionError: No inf checks were recorded for this optimizer.`——**报错在 `GradScaler`，真因在梯度**：那句只是"优化器一个梯度都没收到"，因为没有任何参数 `requires_grad=True`。

**怎么一眼判断 adapter 到底挂没挂**——拿总参数量做减法：

| 数 | 值 | 含义 |
|---|---|---|
| 基座 | 7,615,616,512 | = all params − LoRA |
| Unsloth 侧 `all params` | **7,655,986,688** | 含 40,370,176 个 adapter（验算：40,370,176 ÷ 7,655,986,688 = **0.5273%**，正是日志里的 trainable%） |
| 原生侧 `all params` | **7,655,986,688** | **同一个数** → adapter 也挂上了，只是全被冻住 |

**真因：TRL 的 `SFTTrainer` 会替 4-bit 模型再补一次 PEFT 预处理，而这一步的语义恰恰是"先把所有参数冻住"。**

三处源码，按调用顺序看：

| # | 位置 | 做了什么 |
|---|---|---|
| 1 | `trl/trainer/sft_trainer.py:693` | `if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)): model = prepare_peft_model(model, peft_config, args)` —— **只要传进去的模型已经是 `PeftModel`，就会走这里** |
| 2 | `trl/models/utils.py:536` | `if is_qlora and not is_sharded_qlora: model = prepare_model_for_kbit_training(...)` —— 原生侧经 transformers 的 bnb 量化器加载，模型带着 `is_loaded_in_4bit = True`（`transformers/quantizers/quantizer_bnb_4bit.py:330`），条件成立 |
| 3 | `peft/utils/other.py:193` | `for name, param in model.named_parameters(): param.requires_grad = False` —— **无条件冻住全部参数** |

第 3 步的契约是"**先调它，再调 `get_peft_model`**"——`get_peft_model` 随后会把 adapter 放开（等价于 `_mark_only_adapters_as_trainable`）。而我们的 native 分支原本**自己先 `get_peft_model` 了**，于是 TRL 后来这一冻，把 adapter 一起冻死 → `trainable params: 0`。

**修法：原生侧不再自己挂 adapter，改为把 `LoraConfig` 作为 `peft_config` 交给 `SFTTrainer`**，让它按自己的顺序走（冻基座 → 挂 adapter → adapter 自动可训练）。4.3 里 native 分支末尾的 `return model, tokenizer, peft_config` 就是这个意思；`build_trainer()` 只是把它转到 `SFTTrainer(peft_config=...)`。因为挂 adapter 的时机移到了 trainer 构造过程中，**可训练参数量的断言也必须跟着挪到 `build_trainer()` 之后**（数 `trainer.model`）。

**为什么这条坑只在原生侧发作**：那个分支的触发条件就是上面第 2 步的两个事实——"模型自称 4-bit" 以及"首个 4-bit 参数不在 cpu/meta 上"。Unsloth 侧实测没中招：同一段 `build_trainer`，三次干净运行都正常跑满 60 步、计数 40,370,176（**真被冻住的话，训练会立刻以 `No inf checks were recorded` 收场，不可能跑完**）。4.3 里的 `4-bit 判定 :` 那行诊断，把这两个事实都打了出来，两侧日志一比就有答案——不必去猜 Unsloth 加载器内部做了什么。

> 备选路线（没采用）：保留"自己 `get_peft_model`"，在构造完 trainer 之后再补一次解冻。它能跑（`Trainer` 的优化器是 `train()` 时才建的），但本质是跟 TRL 的既定顺序对抗，不如让 TRL 自己挂干净。

**判据**：4.3 在构造 trainer 之后断言 `可训练参数量 == 40,370,176`，两侧一致才放行。这句断言通过，就不会再撞上 `No inf checks were recorded`。

**⑤ 其他限制**

- **`Qwen/Qwen2.5-7B` 是 fp16 权重（约 15GB）**，两个 notebook 各自要下载一次，比用 Unsloth 预量化仓慢。这是为"同一份权重"付的代价。若确实嫌慢，可以两侧**都**改成 `unsloth/Qwen2.5-7B-bnb-4bit`（原生侧也能加载），但**绝不允许只改一侧**。
- **单卡对比**，不含多卡 / DeepSpeed 场景。
- **Colab 免费档的 GPU 型号会变**，跨天数据不要直接横向比较。

---

## 五、结论

1. **Unsloth 的加速来自底层换内核，不是 API 封装**：封装只为好用，速度来自 Triton 内核与手动反向传播。
2. **对已会原生 HF 微调的人，实际增量很小**：
   - 2 个函数（`FastLanguageModel.from_pretrained` / `get_peft_model`）
   - 1 个参数（`use_gradient_checkpointing="unsloth"`）
   - 1 项新能力（`save_pretrained_gguf` 直接导出 GGUF）
   - 其余（`SFTTrainer`、数据、训练循环、保存）本来就会
3. **本文档只做对照**。训练步骤见 `Unsloth微调实战指南.md`，原理推导见 `LoRA与QLoRA微调大语言模型完整指南.md`。

---

## 附：两侧源码位置

| 侧 | 文件 |
|---|---|
| 原生 HF（基准） | `llm-lora-qlora-finetuning-guide/finetune_llm_with_lora_and_qlora.py` |
| 原生 HF（QLoRA 独立版） | `llm-lora-qlora-finetuning-guide/qlora_finetune_opt.py` |
| Unsloth | `unsloth-finetuning-guide/finetune_basic.py` |
| Unsloth 指南 | `Unsloth微调实战指南.md` |
| 性能基准代码（两侧共用） | **就是本文档 4.3**（内嵌，全仓库无独立 `.py`） |
| 基准跑法、口径、降档预案与结果模板 | 本文档 4.1–4.8 |
