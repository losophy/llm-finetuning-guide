"""模型保存脚本"""

from unsloth import FastLanguageModel


def save_lora_adapter(model, tokenizer, output_dir="./qwen2.5-lora-adapter"):
    """保存 LoRA adapter"""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter 已保存到: {output_dir}")


def push_to_hub(model, repo_name="your-username/qwen2.5-finetuned"):
    """推送到 HuggingFace Hub"""
    model.push_to_hub(repo_name)
    print(f"模型已推送到: {repo_name}")


def load_lora_adapter(model_name="unsloth/Qwen2.5-7B-bnb-4bit",
                      adapter_dir="./qwen2.5-lora-adapter"):
    """加载已保存的 LoRA adapter"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
    )
    model.load_adapter(adapter_dir)
    return model, tokenizer


if __name__ == "__main__":
    # 示例：保存模型
    # save_lora_adapter(model, tokenizer)

    # 示例：推送到 Hub
    # push_to_hub(model)

    print("模型保存脚本")
