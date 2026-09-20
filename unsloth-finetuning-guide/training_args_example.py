"""训练参数配置示例"""

from transformers import TrainingArguments


def get_training_args(output_dir="./qwen2.5-finetuned"):
    """获取训练参数配置"""

    # 标准配置
    training_args = TrainingArguments(
        # 批次大小
        per_device_train_batch_size=2,      # 根据 VRAM 调整
        gradient_accumulation_steps=4,      # 模拟更大批次（有效批次=2*4=8）

        # 学习率
        learning_rate=2e-4,                 # LoRA 标准起点
        lr_scheduler_type="cosine",         # 余弦退火
        warmup_ratio=0.03,                  # 3% 预热

        # 训练轮次
        num_train_epochs=3,                 # 避免过拟合（1-3轮推荐）

        # 精度
        fp16=False,                         # 根据 GPU 支持
        bf16=True,                          # 推荐

        # 其他
        gradient_checkpointing=True,        # 节省 VRAM
        optim="adamw_8bit",                 # 8-bit 优化器

        output_dir=output_dir,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
    )

    return training_args


def get_8gb_vram_args(output_dir="./qwen2.5-finetuned"):
    """8GB 显存优化配置"""
    training_args = TrainingArguments(
        per_device_train_batch_size=1,      # 8GB 必须设为 1
        gradient_accumulation_steps=8,      # 等效批量仍是 8
        max_seq_length=1024,                # 序列长度减半

        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=3,

        fp16=not False,  # 8GB 卡用 fp16
        bf16=False,

        gradient_checkpointing=True,
        optim="adamw_8bit",

        output_dir=output_dir,
        logging_steps=10,
        save_steps=100,
    )

    return training_args


# 不同硬件的推荐配置
HARDWARE_CONFIGS = {
    "6-8GB": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "r": 16,
        "max_seq_length": 1024,
    },
    "12-16GB": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "r": 16,
        "max_seq_length": 2048,
    },
    "24GB": {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "r": 32,
        "max_seq_length": 2048,
    },
    "24GB+": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "r": 16,
        "max_seq_length": 2048,
    },
}


if __name__ == "__main__":
    args = get_training_args()
    print("标准训练参数:")
    print(f"  batch_size: {args.per_device_train_batch_size}")
    print(f"  grad_accum: {args.gradient_accumulation_steps}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  epochs: {args.num_train_epochs}")
