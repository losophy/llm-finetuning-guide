"""数据集准备脚本"""

from datasets import load_dataset

# 默认数据集（HuggingFace Hub，Alpaca 格式）
# 英文通用：yahma/alpaca-cleaned   中文指令：shibing624/alpaca-zh
DEFAULT_DATASET = "yahma/alpaca-cleaned"
# 先小规模跑通，确认无误后再调大（指南"100 条精品"原则）
DEFAULT_MAX_SAMPLES = 1000


def load_from_hub(dataset_name=DEFAULT_DATASET, max_samples=DEFAULT_MAX_SAMPLES):
    """从 HuggingFace Hub 加载公开数据集（默认 Alpaca 格式）

    中国大陆网络可先设置镜像：export HF_ENDPOINT=https://hf-mirror.com
    """
    dataset = load_dataset(dataset_name, split="train")
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    return dataset


def convert_sharegpt(sample):
    """把 ShareGPT 多轮对话（role/content 或 from/value）转成 Alpaca 三字段"""

    def pick(roles):
        for turn in sample["conversations"]:
            role = turn.get("role") or turn.get("from")
            if role in roles:
                return turn.get("content") or turn.get("value") or ""
        return ""

    return {
        "instruction": pick(("user", "human")),
        "input": "",
        "output": pick(("assistant", "gpt")),
    }


def load_from_jsonl(path):
    """从本地 JSONL 加载，自动识别 Alpaca / ShareGPT 两种格式"""
    dataset = load_dataset("json", data_files=path, split="train")
    columns = dataset.column_names
    if "instruction" in columns and "output" in columns:
        return dataset
    if "conversations" in columns:
        return dataset.map(convert_sharegpt, remove_columns=columns)
    raise ValueError(f"无法识别的数据格式，字段为: {columns}")


def prepare_datasets(source=None, test_size=0.2, seed=42):
    """准备训练集/验证集

    source=None  → 加载 HuggingFace 默认数据集（需要联网）
    source=路径  → 加载本地 JSONL（Alpaca 或 ShareGPT 格式均可）
    """
    dataset = load_from_jsonl(source) if source else load_from_hub()
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


if __name__ == "__main__":
    train_dataset, eval_dataset = prepare_datasets()
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    print("示例数据:")
    print(train_dataset[0])
