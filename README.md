# LLM LoRA/QLoRA 微调学习笔记

记录大语言模型 LoRA 与 QLoRA 微调的学习过程，含正确学习路线、实战资源清单与关键心法。

## 学习路径（原理 → 纯代码 → 工业化封装 → 评估部署）

```
第一阶段：死磕原理 —— 本地中文完整指南《LoRA与QLoRA微调大语言模型完整指南.md》（machinelearningplus 翻译版）
   ↓  先逐行手敲 LoRA 代码
第二阶段：用 Unsloth "受虐" —— 纯代码框架跑通 Qwen2.5 微调
   ↓  频繁报错才有价值，每解决一个报错，对 PyTorch / HuggingFace 的理解就深一层
第三阶段：LlamaFactory 工业化封装（配套《ai-agents-from-zero》第 31 章）+ 蒸馏实战
   ↓  此时再看 Web UI，会有"一览众山小"的通透感
第四阶段：微调效果评估与部署（《ai-agents-from-zero》第 32 章）—— 单条验证 → 批量评分 → 导出合并 → 独立加载
```

### 执行清单

| 阶段 | 核心动作 | 产出物 |
|------|----------|--------|
| 阶段一（原理） | 手敲《LoRA与QLoRA微调大语言模型完整指南.md》里的 LoRA 代码（逐行抄，不复制粘贴） | 能画出 LoRA 的原理推导草图，能用大白话讲清"为什么 QLoRA 能把 7B 模型塞进 8G 显存" |
| 阶段二（纯代码） | 在 Unsloth 上跑通 Qwen2.5 微调，换 3 种不同数据集（如阿里天池中文问答） | 至少 3 个不同风格的 Adapter 权重文件 |
| 阶段三（工业化） | 用 LlamaFactory 复现阶段二的结果（对照《ai-agents-from-zero》第 31 章），对比两者速度差异 | 理解工业化框架如何封装底层细节 |
| 阶段四（评估部署） | 按《ai-agents-from-zero》第 32 章做单条验证 → 批量预测与评分 → 合并导出、独立加载 | 一份可复现的评测结果 |
| 蒸馏实战 | 参考 LLM_Optimization，用蒸馏把 7B 模型的知识灌进 0.5B 小模型 | 一个能在笔记本 CPU 上跑起来的蒸馏小模型 |

## 项目清单（按路径重排优先级）

### 🔥 第一优先级：原理教程 + 脚手架

| 项目 | Stars | 核心技术 | 用法 |
|------|-------|----------|------|
| [machinelearningplus - LoRA & QLoRA（中文完整指南）](LoRA与QLoRA微调大语言模型完整指南.md) | — | LoRA/QLoRA 完整代码 | **入门必看**。本仓库含配套脚本（`llm-lora-qlora-finetuning-guide/`），可直接运行。逐行抄写代码，重点关注 `LoraConfig` 里的 `r`（秩）、`alpha`、`target_modules` 怎么影响显存。含内存计算、常见错误 |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 63K+ | LoRA/QLoRA 加速 | **纯代码主框架**。250+ Notebook 示例，必须手动处理 JSON 数据集、手动拆分训练/验证集——痛苦但有用。加速 2-5 倍、显存减半，消费级 GPU 可跑。8GB 本机 / Colab T4 跑法见《Unsloth微调实战指南.md》开头「运行方式」一节。配套代码见 `unsloth-finetuning-guide/` |

### 🔧 第二优先级：工业化框架 + 蒸馏

| 项目 | Stars | 核心技术 | 用法 |
|------|-------|----------|------|
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 71K+ | LoRA/QLoRA/全参/DPO/RLHF | **打通原理、跑通纯代码之后再打开**。微调框架之王，Web UI 零代码，100+ 模型。用来复现 Unsloth 结果、对比封装差异。配套中文实战教程见《ai-agents-from-zero》第 31 章 |
| [didilili/ai-agents-from-zero —— 微调篇（第 28–33 章）](https://github.com/didilili/ai-agents-from-zero) | — | LLaMA-Factory 实战 / 评估部署 / 显存优化 | **中文实战教程，微调篇与本地学习路径完全重合**。29 数据准备与对话模板、30 训练原理与高效微调（LoRA、QLoRA、显存算例）、31 LLaMA-Factory 环境搭建与微调实战（对应阶段三）、32 微调效果评估与模型部署（对应阶段四）、33 显存优化与多卡训练（对应 8GB 显存约束）。克隆：`git clone git@github.com:didilili/ai-agents-from-zero.git` |
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

## 微调篇（第 28–33 章）直达

摘自开源中文教程 [ai-agents-from-zero](https://github.com/didilili/ai-agents-from-zero)（[在线阅读](https://didilili.github.io/ai-agents-from-zero/#/)）：

- [28 大模型微调概述与整体流程](https://github.com/didilili/ai-agents-from-zero/blob/main/28-大模型微调概述与整体流程.md)
- [29 微调数据准备与对话模板](https://github.com/didilili/ai-agents-from-zero/blob/main/29-微调数据准备与对话模板.md)
- [30 模型训练原理与高效微调](https://github.com/didilili/ai-agents-from-zero/blob/main/30-模型训练原理与高效微调.md)
- [31 LLaMA-Factory 环境搭建与微调实战](https://github.com/didilili/ai-agents-from-zero/blob/main/31-LLaMA-Factory环境搭建与微调实战.md)
- [32 微调效果评估与模型部署](https://github.com/didilili/ai-agents-from-zero/blob/main/32-微调效果评估与模型部署.md)
- [33 微调显存优化与多卡训练](https://github.com/didilili/ai-agents-from-zero/blob/main/33-微调显存优化与多卡训练.md)

## 本仓库代码文件

### Unsloth 微调代码（`unsloth-finetuning-guide/`）

| 文件 | 用途 |
|------|------|
| `verify_installation.py` | 验证 PyTorch、CUDA、GPU 环境 |
| `prepare_data.py` | 数据集准备与格式化 |
| `finetune_basic.py` | Qwen2.5 QLoRA 微调主脚本 |
| `training_args_example.py` | 训练参数配置（含 8GB 显存优化） |
| `save_model.py` | 保存 LoRA adapter |
| `export_gguf.py` | 导出 GGUF 格式（Ollama 部署） |

## 两个"隐形的雷"

1. **多卡并行 / DeepSpeed ZeRO（第 33 章）**：单卡 8GB 阶段先跳过，等有多卡环境（或租多卡云实例）再看，别被带偏。
2. **量化工具（GPTQ/AWQ）**：直接用 Unsloth 自带的 `save_pretrained_gguf()` 一键转 GGUF 给 Ollama 部署，不必单独去学 GPTQ 源码。

## 关键心法

- **70% 的时间在洗数据，不是写代码**。把 1000 条杂乱数据整理成高质量问答对，比调参难得多。
- **100 条精品数据原则**：先用 100 条高质量、去重、逻辑严密的数据验证流程，再扩展到 1000 条。100 条精品 > 10000 条乱爬脏数据。如果 100 条跑完是负优化，别加数据，先检查有没有前后矛盾的问答对。
- **数据清洗阶段是弃坑高峰期**（洗数据洗到吐），熬过去就超过了 80% 只会跑 Demo 的人。
- **学看 Loss 曲线做决策**：Loss 震荡太大 → 降学习率；验证集 Loss 回升 → 提前停止；跑完 1 个 Epoch Loss=0.01 → 过拟合了。
- **终局能力是"要不要微调"的决策力**：能用 Few-shot 提示词解决的就不烧算力微调，很多时候微调不如 RAG。
- **硬件决定幸福指数**：显存不足会把大量时间耗在折腾显存溢出上。建议花几十块钱租云 GPU（如 AutoDL 的 A10），把精力省下来看 Loss 曲线。24G 显卡（3090/4090）可微调 7B 参数模型。
- **试错比看书重要**：显卡能点亮就直接跑 Demo，遇红字报错再搜方案，比啃完《深度学习》再碰代码快 10 倍。
