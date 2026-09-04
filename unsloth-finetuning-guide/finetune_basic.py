"""基础微调脚本 - Qwen2.5 QLoRA 微调示例"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def load_model(model_name="unsloth/Qwen2.5-7B-bnb-4bit", max_seq_length=2048):
    """加载模型（QLoRA 4-bit）"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # 自动检测
        load_in_4bit=True,
    )
    return model, tokenizer


def configure_lora(model):
    """配置 LoRA 参数"""
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth 优化
    )
    return model


def format_prompt(sample):
    """格式化 Alpaca 格式数据"""
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""


def create_trainer(model, tokenizer, train_dataset, eval_dataset,
                   max_seq_length=2048, output_dir="./qwen2.5-finetuned"):
    """创建训练器"""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_steps=100,
            optim="adamw_8bit",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            evaluation_strategy="steps",
            eval_steps=100,
        ),
    )
    return trainer


def train(trainer, output_dir="./qwen2.5-finetuned-final"):
    """开始训练并保存模型"""
    trainer.train()
    trainer.save_model(output_dir)
    return trainer


if __name__ == "__main__":
    # 加载模型
    model, tokenizer = load_model()

    # 配置 LoRA
    model = configure_lora(model)

    # 这里需要准备数据集
    # from prepare_data import prepare_datasets
    # train_dataset, eval_dataset = prepare_datasets()

    # 创建训练器
    # trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset)

    # 开始训练
    # train(trainer)

    print("模型加载完成，请准备数据集后运行训练")
