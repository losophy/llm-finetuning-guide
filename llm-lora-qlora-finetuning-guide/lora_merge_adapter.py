import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 模型配置
model_name = "facebook/opt-125m"
adapter_save_path = "./opt-125m-lora-adapter"

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
tokenizer = AutoTokenizer.from_pretrained(adapter_save_path)
merged_model.save_pretrained("./opt-125m-merged")
tokenizer.save_pretrained("./opt-125m-merged")
print("合并后的模型已保存。")
