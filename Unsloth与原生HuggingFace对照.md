# Unsloth 与原生 HuggingFace 对照

> **这份文档解决什么问题**：你手写过原生 HuggingFace/PEFT 的 LoRA/QLoRA 全流程，也已经跑通了 Unsloth。这份文档只回答两件事——**Unsloth 到底替掉了哪些代码**、**它相对原生 HF 能快多少**。
>
> 它**不教微调**。训练步骤见 `Unsloth微调实战指南.md` 第四章；原理推导见 `LoRA与QLoRA微调大语言模型完整指南.md`。
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

条件不对齐，数字就没有意义。两侧必须完全一致：

- **同一基座模型**（两侧用同一份权重）
- **同一份数据、同一采样量**
- **同一套超参**：batch_size、gradient_accumulation_steps、max_length、learning_rate、r / lora_alpha
- **用 `max_steps` 固定步数**（例如 60 步），**不要用 epochs**——否则两侧数据量不同就不可比
- 两侧都**关闭 eval**
- **原生侧不要手动装 Flash Attention 2**，用 PyTorch 默认的 SDPA 即可（T4 上装 FA2 麻烦，且会改变对比口径）

### 4.2 采集代码（两侧共用）

基准脚本：`benchmark-comparison/benchmark_qwen7b.py`。**两侧是同一个文件**——顶部 `SIDE` 一行切换 `"unsloth"` / `"native"`，数据、超参、采集代码物理共用，只有 `load_model()` 分叉。下面这段是该脚本的原文：

```python
import torch
torch.cuda.reset_peak_memory_stats()          # 必须在 train 之前调用

res = trainer.train()

print("耗时(s)      :", res.metrics["train_runtime"])
print("每秒样本数   :", res.metrics.get("train_samples_per_second"))
print("峰值显存(GB) :", torch.cuda.max_memory_allocated() / 1e9)
print("保留显存(GB) :", torch.cuda.max_memory_reserved() / 1e9)
```

> **口径注意**：`reset_peak_memory_stats()` 在 `train()` **之前**调用，所以报的是**训练阶段峰值，不含模型加载**。填表时别当成整机峰值。

### 4.3 两侧怎么跑

在 Colab / Kaggle 各建**两个独立 notebook**，跑一次只跑一侧：

```python
SIDE = "unsloth"    # 另一个 notebook 填 "native"，其余一字不改
```

- **必须独立 notebook（或独立进程）**：`import unsloth` 会在导入时全局改写 HF 的模型实现，同一会话连跑两侧，原生侧数字直接作废。
- **两侧用同一格 install cell**，打印出的库版本（torch / transformers / trl / peft / bitsandbytes）必须一致。
- **两侧必须同一型号 GPU**（T4 与 L4 不可混在一张表里）；脚本会打印 `torch.cuda.get_device_name(0)`，跑之前确认。
- 统一口径：基座统一 `Qwen/Qwen2.5-7B`（各自加载时量化，不用预量化仓）；`target_modules` 统一 7 个；精度统一 fp16（T4 无 bf16）；原生侧不装 Flash Attention 2；跑 2~3 次取平均。
- 完整的九条口径、降档预案与结果记录表见 `benchmark-comparison/README.md`。

### 4.4 结果表（待填）

| 指标 | Unsloth | 原生 HF | 倍数 / 差异 |
|---|---|---|---|
| 训练耗时（s） | | | |
| 峰值显存（GB） | | | |
| 每秒样本数 | | | |
| 可训练参数量 | | | |

> **填表前先自检**：两侧"可训练参数量"必须**完全相同**。不一致说明 `target_modules` / `r` / `alpha` 有一侧漂了，这组数字作废。

**环境脚注（每次填表都要写）**：GPU 型号 · 库版本（torch / transformers / trl / peft / bitsandbytes）· `max_steps` · `max_length` · batch × accum · `target_modules` · 跑了几次是否取平均。模板见 `benchmark-comparison/README.md` 第五节。

### 4.5 官方公开数字（**非本人实测，仅作量级参考**）

> 来源：HuggingFace 官方博客 https://huggingface.co/blog/unsloth-trl
> 对比基线：**"HF + Flash Attention 2"**；测试环境：A100 40GB 与免费 Colab T4。

| 环境 | 模型 / 数据 | 加速 | 显存降幅 |
|---|---|---|---|
| A100 40GB | Llama-2 7b / Slim Orca | 1.87× | -39.3% |
| A100 40GB | Mistral 7b / Slim Orca | 1.88× | -65.9% |
| A100 40GB | Tiny Llama 1.1b / Alpaca | 2.74× | -57.8% |
| **免费 Colab T4** | Llama-2 7b / OASST | **1.95×** | **-43.3%** |
| 免费 Colab T4 | Mistral 7b / Alpaca | 1.56× | -13.7% |

**版本注意**：这批数据来自 transformers 4.36 / PyTorch 2.1.1 时期，与当前版本（TRL 1.12 / transformers 5.16）不同，仅作量级参考。要形成自己的结论，请以 4.4 的实测数字为准。

### 4.6 三个坑

1. **T4 上原生 HF 跑 7B QLoRA 可能 OOM**——原生没有内核优化，显存占用明显更高（官方数据里降幅在 12%~74% 之间，视模型而定）。**若原生侧 OOM，把两侧一起降到同一档小模型**（不能只降一边，否则失去可比性）。
2. **单次数字有噪声**——同条件跑 2~3 次取平均更稳。
3. **公平性**——两侧都不要手动装 FA2；原生侧用默认 SDPA。

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
| 性能基准脚本（两侧共用） | `benchmark-comparison/benchmark_qwen7b.py` |
| 基准跑法与结果模板 | `benchmark-comparison/README.md` |
