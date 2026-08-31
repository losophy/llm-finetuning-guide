import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
print(f"模型权重精度: {next(model_qlora.parameters()).dtype}")
