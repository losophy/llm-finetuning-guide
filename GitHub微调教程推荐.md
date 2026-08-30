# GitHub 微调教程推荐（匹配JD要求 · 已按实战路线重排）

> 针对「后端开发高级工程师」JD中的微调要求：LoRA/QLoRA、蒸馏、轻量化优化、Python/PyTorch
>
> **学习周期定位：2~3 个月**（针对"能独立调参、稳定提升效果"的工程目标）
>
> ⚠️ **核心修正**：原路径"LlamaFactory → 端到端 → 原理"是经典工程师陷阱。先玩 Web UI、最后看原理，2个月后你会变成"只会点按钮的调参侠"——面试官一问"LoRA 的秩为什么设 16 而不是 64"就懵了。**必须倒过来：原理 → 纯代码框架 → 工业化封装。**

---

## 正确的学习路径（"倒三角"打法，10周足矣）

```
第一阶段（第1~2周）：死磕原理 — machinelearningplus 教程
   ↓  千万别一上来就开 LlamaFactory Web UI！先逐行手敲 LoRA 代码
第二阶段（第3~6周）：用 Unsloth "受虐" — 纯代码框架跑通 Qwen2.5 微调
   ↓  频繁报错才有价值，每解决一个报错对 PyTorch/HF 的理解就深一层
第三阶段（第7~10周）：LlamaFactory 工业化封装 + 蒸馏实战
   ↓  此时再看 Web UI 会有"一览众山小"的通透感
第11周起：腾讯云 Agent 教程（可选） / 面试题背诵
```

### 执行清单（照着打勾）

| 周数 | 核心动作 | 产出物 |
|------|----------|--------|
| 1-2周 | 手敲 machinelearningplus 的 LoRA 代码（逐行抄，不复制粘贴） | 能画出 LoRA 的原理推导草图，能用大白话讲清"为什么 QLoRA 能把 7B 模型塞进 8G 显存" |
| 3-6周 | 在 Unsloth 上跑通 Qwen2.5 微调，换 3 种不同数据集（如阿里天池中文问答） | 至少 3 个不同风格的 Adapter 权重文件 |
| 7-8周 | 用 LlamaFactory 复现第 6 周的结果，对比两者速度差异 | 理解工业化框架如何封装底层细节 |
| 9-10周 | 参考 LLM_Optimization，用蒸馏把 7B 模型的知识灌进 0.5B 小模型 | 一个能在笔记本 CPU 上跑起来的蒸馏小模型 |
| 11-12周 | 面试题背诵（"LoRA 和 Adapter 的区别"、"蒸馏损失函数怎么写"） | 简历可写"具备端到端微调与蒸馏落地经验" |

---

## 项目清单（按新路径重排优先级）

### 🔥 第一优先级：原理教程 + 脚手架（第1~6周主战场）

| 项目 | Stars | 核心技术 | 用法 |
|------|-------|----------|------|
| [machinelearningplus - LoRA & QLoRA](https://machinelearningplus.com/deep-learning/fine-tuning-llms-lora-qlora-python/) | — | LoRA/QLoRA完整代码 | **第1-2周必看**。打开 Colab 笔记本逐行抄写代码，重点关注 `LoraConfig` 里的 `r`（秩）、`alpha`、`target_modules` 怎么影响显存。60分钟教程，含内存计算、常见错误 |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 63K+ | LoRA/QLoRA加速 | **第3-6周主框架**。纯代码、250+ Notebook 示例，必须手动处理 JSON 数据集、手动拆分训练/验证集——痛苦但有用。加速2-5倍、显存减半，消费级GPU可跑 |

### 🔧 第二优先级：工业化框架 + 蒸馏（第7~10周）

| 项目 | Stars | 核心技术 | 用法 |
|------|-------|----------|------|
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 71K+ | LoRA/QLoRA/全参/DPO/RLHF | **第7周再打开**。微调框架之王，Web UI 零代码，100+模型，JD面试加分项。用来复现 Unsloth 结果、对比封装差异 |
| [hdtinh57/LLM_Optimization](https://github.com/hdtinh57/LLM_Optimization) | 新 | QLoRA→CoT蒸馏→GGUF量化→Ollama部署 | **最贴合JD但⚠️致命预警**：项目新、Star少，千万别直接跑全量代码（大概率报错无人解答）。正确操作是**"偷"它的蒸馏逻辑**——看它怎么用教师模型生成软标签训练学生模型，再移植到 LlamaFactory 里跑 |

### 📚 参考梯队（有余力再看）

| 项目 | Stars | 核心技术 | 说明 |
|------|-------|----------|------|
| [kooroshsajadi/llm-fine-tuning-and-distillation](https://github.com/kooroshsajadi/llm-fine-tuning-and-distillation) | 新 | QLoRA+知识蒸馏+评估 | 模块化设计，含 ROUGE/BLEU 评估，作蒸馏参考 |
| [huangxiaoye6/LLM-tuning](https://github.com/huangxiaoye6/LLM-tuning) | 39 | LoRA/P-tuning/GPTQ/AWQ | **中文**，Qwen3-0.6B实例，含量化操作完整代码 |
| [lailoo/Hello-LLM-FineTuning](https://github.com/lailoo/Hello-LLM-FineTuning) | 中 | LoRA/QLoRA/Adapter/P-Tuning/vLLM部署 | 全栈微调指南，理论+代码 |
| [rmisegal/llm-lora-project](https://github.com/rmisegal/llm-lora-project) | 1 | LoRA 14个渐进任务 | 教学项目，从基础到高级 |
| [AdityaSagarr/LLM-Fine-Tuning](https://github.com/AdityaSagarr/LLM-Fine-Tuning) | 1 | LoRA/QLoRA完整流程 | Colab可跑，Llama-2-7B实例 |
| [technoscripts - 5步微调](https://technoscripts.com/python-fine-tuning-llm/) | — | LoRA/QLoRA 5步流程 | 单GPU实操：数据准备→训练→合并→部署 |
| [腾讯云 - 微调+Agent实战](https://developer.cloud.tencent.com/article/2716375) | — | QLoRA+蒸馏+多工具调用 | ⚠️ **第11周之后再碰**。微调还没学好就加 Agent，梯度爆炸会让你直接弃坑 |

---

## 两个"隐形的雷"

1. **腾讯云教程（Agent）**：涉及智能体和多工具调用，微调基础不牢时不要碰，建议第11周后。
2. **量化工具（GPTQ/AWQ）**：**直接用 Unsloth 自带的 `save_pretrained_gguf()` 一键转 GGUF 给 Ollama 部署**。不要单独去学 GPTQ 源码——那是算法工程师的事，后端开发会调用 API 转换即可。

---

## 关键心法（决定你能不能熬过2~3个月）

- **70% 的时间在洗数据，不是写代码**。把1000条杂乱数据整理成高质量问答对，比调参难得多。
- **100条精品数据原则**：先用100条高质量、去重、逻辑严密的数据验证流程，再扩展到1000条。100条精品 > 10000条乱爬脏数据。如果100条跑完是负优化，别加数据，先检查有没有前后矛盾的问答对。
- **第5周是弃坑高峰期**（洗数据洗到吐），熬过去你就超过80%只会跑Demo的人。
- **学看 Loss 曲线做决策**：Loss 震荡太大→降学习率；验证集 Loss 回升→提前停止；跑完1个Epoch Loss=0.01→过拟合了。
- **终局能力是"要不要微调"的决策力**：能用 Few-shot 提示词解决的就不烧算力微调，很多时候微调不如 RAG。
- **硬件决定幸福指数**：显存<8G 的话，这3个月有1个月在折腾显存溢出。建议花几十块钱租云GPU（如 AutoDL 的 A10），把精力省下来看 Loss 曲线。24G 显卡（3090/4090）可微调7B参数模型。
- **试错比看书重要**：显卡能点亮就直接跑 Demo，遇红字报错再搜方案，比啃完《深度学习》再碰代码快10倍。

---

## JD要求匹配度

| JD要求 | 覆盖项目 |
|--------|----------|
| LoRA/QLoRA微调 | 全部项目 |
| 蒸馏（Distillation） | hdtinh57/LLM_Optimization、kooroshsajadi/llm-fine-tuning-and-distillation、腾讯云教程 |
| 轻量化优化/量化 | hdtinh57/LLM_Optimization、huangxiaoye6/LLM-tuning（GPTQ/AWQ）、Unsloth `save_pretrained_gguf()` |
| Python/PyTorch | 全部项目 |
| HuggingFace生态 | 全部项目（PEFT/Transformers/TRL） |
| 端到端全流程 | hdtinh57/LLM_Optimization（参考逻辑）+ LlamaFactory（实操框架） |
