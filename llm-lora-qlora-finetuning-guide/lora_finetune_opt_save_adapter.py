import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# 数据格式化函数（来自prepare_sft_dataset.py）
def format_instruction(sample):
    return f"### Human: {sample['question']}\n### Assistant: {sample['answer']}"

# 加载数据集
raw_qa_pairs = [
    {"question": "What is LoRA?", "answer": "LoRA is a parameter-efficient fine-tuning method..."},
    {"question": "How does QLoRA work?", "answer": "QLoRA combines 4-bit quantization with LoRA..."},
]

print("自定义数据集格式:")
print(format_instruction(raw_qa_pairs[0])[:200])

# 使用公开数据集（来自prepare_sft_dataset.py）
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
print(f"\n数据集大小: {len(dataset)} 条样本")
print(f"\n预格式化示例 (前 250 个字符):")
print(dataset[0]["text"][:250])

# 模型配置（先用 opt-125m 验证流程，8GB 显卡放不下 Qwen2-7B 的 bf16 权重 ~14GB）
model_name = "facebook/opt-125m"  # 验证流程用；7B 需 QLoRA 4bit（见文件底部注释）
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,  # transformers 5.x 已弃用 torch_dtype
    device_map="auto",
)

# 打印模型信息（来自opt-125m-lora-sandbox.py）
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型: {model_name}")
print(f"总参数: {total_params / 1e6:.1f}M")
print(f"可训练参数 (LoRA 之前): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
print(f"模型精度: {next(model.parameters()).dtype}")

# LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 打印LoRA参数信息
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"\nLoRA配置后:")
print(f"可训练参数: {trainable_params / 1e6:.1f}M ({100 * trainable_params / all_params:.2f}%)")
print(f"总参数: {all_params / 1e6:.1f}M")

# 训练配置
training_args = SFTConfig(
    output_dir="./opt-125m-lora",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_steps=100,
    logging_steps=10,
    bf16=True,
    report_to="none",
    max_length=512,
    dataset_text_field="text",
    loss_type="nll",  # 规避 trl 1.x chunked_nll 与 transformers 5.x Qwen2 forward(partial) 的兼容 bug
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 开始训练
print("\n开始训练...")
trainer.train()
print("训练完成!")

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
