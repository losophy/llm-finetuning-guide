# -*- coding: utf-8 -*-
"""
生成评测集 evalset/test.jsonl (位于脚本所在目录)
用法: python create_testset.py            # 生成默认 30 条
      python create_testset.py --out 自定义路径.jsonl
"""
import argparse
import json
import os

TEST_SET = [
    # ---------- A. 概念解释类 (6/15) ----------
    {"category": "concept", "instruction": "What is LoRA?", "reference": "LoRA is a parameter-efficient fine-tuning method that injects low-rank matrices into frozen model weights, so only a tiny fraction of parameters are trained."},
    {"category": "concept", "instruction": "How does QLoRA work?", "reference": "QLoRA combines 4-bit quantization of the base model with LoRA adapters trained in 16-bit precision, cutting memory usage by about 75 percent."},
    {"category": "concept", "instruction": "What is the difference between LoRA and full fine-tuning?", "reference": "Full fine-tuning updates every weight of the model, while LoRA only updates small injected low-rank matrices and keeps the base model frozen."},
    {"category": "concept", "instruction": "Why do we use quantization when fine-tuning large language models?", "reference": "Quantization reduces the memory footprint of model weights, for example from 16-bit to 4-bit, so larger models fit on limited GPU memory."},
    {"category": "concept", "instruction": "Explain the role of the low-rank matrices in LoRA.", "reference": "The low-rank matrices approximate the weight update. Instead of learning a large matrix W, LoRA learns two small matrices A and B whose product AB represents the change."},
    {"category": "concept", "instruction": "What does it mean to freeze model weights during fine-tuning?", "reference": "Freezing means the base weights are not updated by the optimizer. Only the newly added parameters, such as LoRA matrices, receive gradient updates."},

    # ---------- B. 指令遵循·格式约束类 (6/15) ----------
    {"category": "instruction", "instruction": "List three advantages of LoRA.", "reference": "Three advantages: 1) much lower memory and storage cost; 2) training is fast because few parameters update; 3) adapters are small and easy to swap or deploy."},
    {"category": "instruction", "instruction": "Answer in one sentence: what is a learning rate?", "reference": "The learning rate controls how large a step the optimizer takes when updating parameters."},
    {"category": "instruction", "instruction": "Define overfitting in fewer than 15 words.", "reference": "A model performs well on training data but poorly on new data."},
    {"category": "instruction", "instruction": "Reply with only Yes or No: Is QLoRA a 4-bit quantized way of fine-tuning?", "reference": "Yes."},
    {"category": "instruction", "instruction": "Name two hyperparameters used in fine-tuning. Start your answer with 'Two hyperparameters are'.", "reference": "Two hyperparameters are learning rate and batch size."},
    {"category": "instruction", "instruction": "Complete the sentence: 'Gradient descent is a method to ...'", "reference": "Gradient descent is a method to iteratively minimize a loss function by moving parameters in the direction of the negative gradient."},

    # ---------- C. 判断比较类 (4/10) ----------
    {"category": "compare", "instruction": "Which is faster to train, LoRA or full fine-tuning? Explain briefly.", "reference": "LoRA is faster because it updates only a small number of parameters, so computation and memory are much lower than updating the whole model."},
    {"category": "compare", "instruction": "Is a larger LoRA rank r always better? Justify briefly.", "reference": "No. A larger r increases capacity but also trainable parameters and memory. If r is too large, LoRA approaches full fine-tuning cost with diminishing benefits."},
    {"category": "compare", "instruction": "Compare bf16 and fp16 in two short points.", "reference": "1) bf16 has the same exponent range as fp32, so it is more stable in training; 2) fp16 has less range but more precision on small values, and may need loss scaling."},
    {"category": "compare", "instruction": "Do we need to update the base model weights in LoRA? Answer and explain.", "reference": "No. In LoRA the base weights stay frozen; only the injected low-rank matrices are trained, which is why it is called parameter-efficient."},

    # ---------- D. 改写润色类 (4/10) ----------
    {"category": "rewrite", "instruction": "Rewrite this sentence in a more formal tone: 'The model is very big.'", "reference": "The model has a very large number of parameters."},
    {"category": "rewrite", "instruction": "Rewrite this sentence as a question: 'LoRA saves GPU memory.'", "reference": "Does LoRA save GPU memory?"},
    {"category": "rewrite", "instruction": "Make this text shorter: 'Fine-tuning a large language model requires a lot of GPU memory because every parameter is updated during training.'", "reference": "Fine-tuning large models is memory-hungry because all parameters are updated."},
    {"category": "rewrite", "instruction": "Change this sentence to start with 'By using LoRA': 'We can fine-tune a 7B model on one GPU.'", "reference": "By using LoRA, we can fine-tune a 7B model on one GPU."},

    # ---------- E. 简短建议生成类 (4/10) ----------
    {"category": "advice", "instruction": "Give one tip to reduce GPU memory usage when fine-tuning, and explain why it helps.", "reference": "Use 4-bit quantization (QLoRA). It shrinks base weights from 16-bit to 4-bit, cutting memory by about 75 percent with little quality loss."},
    {"category": "advice", "instruction": "Give an example of a hyperparameter and a typical value used in LoRA.", "reference": "Learning rate, for example 2e-4, is a typical hyperparameter used when training LoRA adapters."},
    {"category": "advice", "instruction": "What is one risk of fine-tuning on a very small dataset? Answer in one sentence.", "reference": "The model may overfit and memorize the few examples instead of learning a general behavior."},
    {"category": "advice", "instruction": "What should you check first when the training loss does not decrease?", "reference": "Check the learning rate and whether gradients are actually flowing to the trainable parameters, for example by verifying that only LoRA weights require gradients."},
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalset", "test.jsonl"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, item in enumerate(TEST_SET):
            row = {"id": i, **item}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已生成 {len(TEST_SET)} 条 -> {args.out}")
    from collections import Counter
    print("类别分布:", dict(Counter(t["category"] for t in TEST_SET)))

if __name__ == "__main__":
    main()
