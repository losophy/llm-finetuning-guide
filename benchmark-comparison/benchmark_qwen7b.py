"""Qwen2.5-7B QLoRA 同条件性能基准：Unsloth vs 原生 HuggingFace

用途：跑出《Unsloth与原生HuggingFace对照.md》第四节 4.4 结果表要填的数字。

怎么跑（Colab / Kaggle 免费 GPU，T4 16GB）：
    1) 先建一个 notebook，运行环境准备 cell（见 README.md）
    2) 把本文件整段粘进下一个 cell
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
    """唯一允许两侧分叉的地方"""
    if SIDE == "unsloth":
        from unsloth import FastLanguageModel  # 延迟导入：只有这一侧才 import
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            dtype=COMPUTE_DTYPE,
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
        return model, tokenizer

    if SIDE == "native":
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, TaskType, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
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
        model.enable_input_require_grads()      # 让梯度到达 LoRA adapter
        model = get_peft_model(model, LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=TARGET_MODULES,
        ))
        return model, tokenizer

    raise ValueError(f'SIDE 只能填 "unsloth" 或 "native"，当前是 {SIDE!r}')


def build_trainer(model, tokenizer, train_dataset):
    """两侧共用同一个训练器配置"""
    from trl import SFTConfig, SFTTrainer
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
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

model, tokenizer = load_model()
train_dataset = build_dataset()
trainer = build_trainer(model, tokenizer, train_dataset)
model.print_trainable_parameters()
print("-" * 60)

# ↓↓↓ 与《Unsloth与原生HuggingFace对照.md》4.2 代码块逐字一致 ↓↓↓
import torch
torch.cuda.reset_peak_memory_stats()          # 必须在 train 之前调用

res = trainer.train()

print("耗时(s)      :", res.metrics["train_runtime"])
print("每秒样本数   :", res.metrics.get("train_samples_per_second"))
print("峰值显存(GB) :", torch.cuda.max_memory_allocated() / 1e9)
print("保留显存(GB) :", torch.cuda.max_memory_reserved() / 1e9)
# ↑↑↑ 与文档 4.2 代码块逐字一致 ↑↑↑
