"""数据集准备脚本"""

from datasets import Dataset


def create_dataset():
    """创建示例数据集"""
    data = {
        "instruction": ["任务1", "任务2", "任务3"],
        "input": ["输入1", "输入2", "输入3"],
        "output": ["输出1", "输出2", "输出3"]
    }
    dataset = Dataset.from_dict(data)
    return dataset


def split_dataset(dataset, test_size=0.2):
    """划分训练集和验证集"""
    split_dataset = dataset.train_test_split(test_size=test_size)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    return train_dataset, eval_dataset


def format_prompt(sample):
    """格式化 Alpaca 格式数据"""
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""


def prepare_datasets():
    """准备训练数据集"""
    dataset = create_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)

    # 应用格式化
    train_dataset = train_dataset.map(lambda x: {"text": format_prompt(x)})
    eval_dataset = eval_dataset.map(lambda x: {"text": format_prompt(x)})

    return train_dataset, eval_dataset


if __name__ == "__main__":
    train_dataset, eval_dataset = prepare_datasets()
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    print(f"示例数据: {train_dataset[0]}")
