import os
from modelscope.hub.snapshot_download import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

model_dir = "D:\\llm-lora-qlora-finetuning-guide\\llm-lora-qlora-finetuning-guide\\Qwen2-7B"
snapshot_dir = os.path.join(model_dir, "models", "Qwen--Qwen2-7B", "snapshots", "master")

if not os.path.exists(snapshot_dir):
    print("从 ModelScope 下载模型...")
    snapshot_download("Qwen/Qwen2-7B", cache_dir=model_dir)
else:
    print("模型已存在，跳过下载")

print(f"从本地加载模型: {snapshot_dir}")
model = AutoModelForCausalLM.from_pretrained(
    snapshot_dir, 
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print("应用 LoRA 之后:")
print(f"  可训练:  {trainable:,}  ({trainable / total * 100:.2f}%)")
print(f"  冻结:     {total - trainable:,}")
model.print_trainable_parameters()
