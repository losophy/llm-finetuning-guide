import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 模型配置
model_name = "facebook/opt-125m"
adapter_save_path = "./opt-125m-lora-adapter"

# 加载 adapter 并快速评估
base = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
fine_tuned = PeftModel.from_pretrained(base, adapter_save_path)
fine_tuned.eval()

tokenizer = AutoTokenizer.from_pretrained(adapter_save_path)

test_prompt = "### Human: Explain the concept of gradient descent.\n### Assistant:"
inputs = tokenizer(test_prompt, return_tensors="pt").to(fine_tuned.device)

with torch.no_grad():
    output_ids = fine_tuned.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
print("回答:", generated)
