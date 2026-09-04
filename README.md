# LLM LoRA/QLoRA 微调学习笔记

记录大语言模型 LoRA 与 QLoRA 微调的学习过程，含正确学习路线、实战资源清单与关键心法。

## 学习路径（原理 → 纯代码 → 工业化封装）

```
第一阶段：死磕原理 —— 本地中文完整指南《LoRA与QLoRA微调大语言模型完整指南.md》（machinelearningplus 翻译版）
   ↓  先逐行手敲 LoRA 代码
第二阶段：用 Unsloth "受虐" —— 纯代码框架跑通 Qwen2.5 微调
   ↓  频繁报错才有价值，每解决一个报错，对 PyTorch / HuggingFace 的理解就深一层
第三阶段：LlamaFactory 工业化封装 + 蒸馏实战
   ↓  此时再看 Web UI，会有"一览众山小"的通透感
（可选延伸）腾讯云 Agent 教程 —— 微调基础打牢之后再碰
```

### 执行清单

| 阶段 | 核心动作 | 产出物 |
|------|----------|--------|
| 阶段一（原理） | 手敲《LoRA与QLoRA微调大语言模型完整指南.md》里的 LoRA 代码（逐行抄，不复制粘贴） | 能画出 LoRA 的原理推导草图，能用大白话讲清"为什么 QLoRA 能把 7B 模型塞进 8G 显存" |
| 阶段二（纯代码） | 在 Unsloth 上跑通 Qwen2.5 微调，换 3 种不同数据集（如阿里天池中文问答） | 至少 3 个不同风格的 Adapter 权重文件 |
| 阶段三（工业化） | 用 LlamaFactory 复现阶段二的结果，对比两者速度差异 | 理解工业化框架如何封装底层细节 |
| 蒸馏实战 | 参考 LLM_Optimization，用蒸馏把 7B 模型的知识灌进 0.5B 小模型 | 一个能在笔记本 CPU 上跑起来的蒸馏小模型 |

## 项目清单（按路径重排优先级）

### 🔥 第一优先级：原理教程 + 脚手架

| 项目 | Stars | 核心技术 | 用法 |
|------|-------|----------|------|
| [machinelearningplus - LoRA & QLoRA（中文完整指南）](LoRA与QLoRA微调大语言模型完整指南.md) | — | LoRA/QLoRA 完整代码 | **入门必看**。本仓库含配套脚本（`llm-lora-qlora-finetuning-guide/`），可直接运行。逐行抄写代码，重点关注 `LoraConfig` 里的 `r`（秩）、`alpha`、`target_modules` 怎么影响显存。含内存计算、常见错误 |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 63K+ | LoRA/QLoRA 加速 | **纯代码主框架**。250+ Notebook 示例，必须手动处理 JSON 数据集、手动拆分训练/验证集——痛苦但有用。加速 2-5 倍、显存减半，消费级 GPU 可跑 |

### 🔧 第二优先级：工业化框架 + 蒸馏

| 项目 | Stars | 核心技术 | 用法 |
|------|-------|----------|------|
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 71K+ | LoRA/QLoRA/全参/DPO/RLHF | **打通原理、跑通纯代码之后再打开**。微调框架之王，Web UI 零代码，100+ 模型。用来复现 Unsloth 结果、对比封装差异 |
| [hdtinh57/LLM_Optimization](https://github.com/hdtinh57/LLM_Optimization) | 新 | QLoRA → CoT 蒸馏 → GGUF 量化 → Ollama 部署 | ⚠️ **致命预警**：项目新、Star 少，千万别直接跑全量代码（大概率报错无人解答）。正确操作是"偷"它的蒸馏逻辑——看它怎么用教师模型生成软标签训练学生模型，再移植到 LlamaFactory 里跑 |

### 📚 参考梯队（有余力再看）

| 项目 | Stars | 核心技术 | 说明 |
|------|-------|----------|------|
| [kooroshsajadi/llm-fine-tuning-and-distillation](https://github.com/kooroshsajadi/llm-fine-tuning-and-distillation) | 新 | QLoRA + 知识蒸馏 + 评估 | 模块化设计，含 ROUGE/BLEU 评估，作蒸馏参考 |
| [huangxiaoye6/LLM-tuning](https://github.com/huangxiaoye6/LLM-tuning) | 39 | LoRA/P-tuning/GPTQ/AWQ | **中文**，Qwen3-0.6B 实例，含量化操作完整代码 |
| [lailoo/Hello-LLM-FineTuning](https://github.com/lailoo/Hello-LLM-FineTuning) | 中 | LoRA/QLoRA/Adapter/P-Tuning/vLLM 部署 | 全栈微调指南，理论 + 代码 |
| [rmisegal/llm-lora-project](https://github.com/rmisegal/llm-lora-project) | 1 | LoRA 14 个渐进任务 | 教学项目，从基础到高级 |
| [AdityaSagarr/LLM-Fine-Tuning](https://github.com/AdityaSagarr/LLM-Fine-Tuning) | 1 | LoRA/QLoRA 完整流程 | Colab 可跑，Llama-2-7B 实例 |
| [technoscripts - 5步微调](https://technoscripts.com/python-fine-tuning-llm/) | — | LoRA/QLoRA 5 步流程 | 单 GPU 实操：数据准备 → 训练 → 合并 → 部署 |
| [腾讯云 - 微调+Agent实战](https://developer.cloud.tencent.com/article/2716375) | — | QLoRA + 蒸馏 + 多工具调用 | ⚠️ **放到学习后期再碰**。微调还没学好就加 Agent，梯度爆炸会让你直接弃坑 |

## 两个"隐形的雷"

1. **腾讯云教程（Agent）**：涉及智能体和多工具调用，微调基础不牢时不要碰，放到学习后期。
2. **量化工具（GPTQ/AWQ）**：直接用 Unsloth 自带的 `save_pretrained_gguf()` 一键转 GGUF 给 Ollama 部署，不必单独去学 GPTQ 源码。

## 关键心法

- **70% 的时间在洗数据，不是写代码**。把 1000 条杂乱数据整理成高质量问答对，比调参难得多。
- **100 条精品数据原则**：先用 100 条高质量、去重、逻辑严密的数据验证流程，再扩展到 1000 条。100 条精品 > 10000 条乱爬脏数据。如果 100 条跑完是负优化，别加数据，先检查有没有前后矛盾的问答对。
- **数据清洗阶段是弃坑高峰期**（洗数据洗到吐），熬过去就超过了 80% 只会跑 Demo 的人。
- **学看 Loss 曲线做决策**：Loss 震荡太大 → 降学习率；验证集 Loss 回升 → 提前停止；跑完 1 个 Epoch Loss=0.01 → 过拟合了。
- **终局能力是"要不要微调"的决策力**：能用 Few-shot 提示词解决的就不烧算力微调，很多时候微调不如 RAG。
- **硬件决定幸福指数**：显存不足会把大量时间耗在折腾显存溢出上。建议花几十块钱租云 GPU（如 AutoDL 的 A10），把精力省下来看 Loss 曲线。24G 显卡（3090/4090）可微调 7B 参数模型。
- **试错比看书重要**：显卡能点亮就直接跑 Demo，遇红字报错再搜方案，比啃完《深度学习》再碰代码快 10 倍。
