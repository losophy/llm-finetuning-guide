import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "facebook/opt-125m"  # 生产环境换成 "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # OPT 没有单独的 pad token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型: {model_name}")
print(f"总参数: {total_params / 1e6:.1f}M")
print(f"可训练参数 (LoRA 之前): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
print(f"模型精度: {next(model.parameters()).dtype}")