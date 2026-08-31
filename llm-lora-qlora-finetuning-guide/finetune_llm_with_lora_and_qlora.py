# 完整代码：使用 LoRA 和 QLoRA 在 Python 中微调 LLM
# pip install torch transformers peft trl bitsandbytes datasets accelerate
# Python 3.9+ | 需要 6+ GB 显存的 GPU

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
    model=model, processing_class=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        output_dir="./opt-125m-lora", num_train_epochs=1,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, max_length=512,
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
    model=model_q, processing_class=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        output_dir="./opt-125m-qlora", num_train_epochs=1,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, bf16=True, max_length=512,
        dataset_text_field="text", optim="paged_adamw_32bit", report_to="none",
    ),
).train()

print("流水线全部完成。")
