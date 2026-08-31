import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
print(f"数据集大小: {len(dataset)} 条样本")

# QLoRA 第 1 步：配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                 # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,     # 计算时反量化为 bfloat16
    bnb_4bit_use_double_quant=True,            # 双重量化：每个参数节省约 0.37 bit
)

# QLoRA 第 2 步：带量化加载——与 LoRA 相比唯一不同的那一行
model_name = "facebook/opt-125m"
model_qlora = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 加载量化模型后，这两行是必须的
model_qlora.config.use_cache = False        # 与梯度检查点不兼容
model_qlora.enable_input_require_grads()    # 让梯度能到达 LoRA adapter

print("模型以 4-bit 精度加载")

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
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

qlora_args = SFTConfig(
    output_dir="./opt-125m-qlora",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,             # trl 1.12 已移除 warmup_ratio，改用 warmup_steps（约 616 步的 3%）
    bf16=True,
    max_length=512,              # trl 1.12 中 max_seq_length 已更名为 max_length
    dataset_text_field="text",
    loss_type="nll",             # 与 lora_finetune_opt.py 一致，规避 trl 1.x chunked_nll 兼容问题
    optim="paged_adamw_32bit",   # 分页优化器——防止训练尖峰时 OOM
    report_to="none",
)

qlora_trainer = SFTTrainer(
    model=model_qlora,
    args=qlora_args,
    train_dataset=dataset,
    processing_class=tokenizer,  # trl 1.12 中 tokenizer 参数已更名为 processing_class
)
qlora_trainer.train()
print("QLoRA 训练完成!")
