from datasets import load_dataset

# 把你自己的原始数据转换成 SFTTrainer 格式
# 每个 text 字符串把提示词 + 回答合并为一个字段
raw_qa_pairs = [
    {"question": "What is LoRA?", "answer": "LoRA is a parameter-efficient fine-tuning method..."},
    {"question": "How does QLoRA work?", "answer": "QLoRA combines 4-bit quantization with LoRA..."},
]

def format_instruction(sample):
    return f"### Human: {sample['question']}\n### Assistant: {sample['answer']}"

# 展示格式化后的文本长什么样
print("自定义数据集格式:")
print(format_instruction(raw_qa_pairs[0])[:200])

# 本教程使用一个预格式化好的公开数据集
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
print(f"\n数据集大小: {len(dataset)} 条样本")
print(f"\n预格式化示例 (前 250 个字符):")
print(dataset[0]["text"][:250])