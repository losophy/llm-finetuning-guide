# Unsloth 微调效果 A-B 对照指南

> **用法**：动手做——adapter 训完、GGUF 也能跑了之后，用这一章回答一个问题：**它到底改变了什么？**
>
> **定位**：接在《Unsloth微调实战指南.md》**阶段二**之后。测的是 **LoRA adapter 的净效应**（同一个基座，挂 vs 不挂），**不是**"模型好不好"。
>
> **前置**：已有 `qwen2.5-finetuned-final`（指南 4.1 产出）。**只需 Kaggle，不需要本地环境。**
>
> **产出**：一份 `ab_raw.json` + 两张结果表 + 一段结论。

---

## 一、⚠️ 先排除一个错误设计

**别拿 `ollama pull qwen2.5:7b` 当对照组**——它和你的微调版不是同一个起跑线：

| | 你的微调版 | Ollama `qwen2.5:7b` |
|---|---|---|
| 基座 | `unsloth/Qwen2.5-7B-bnb-4bit` = **Base**（指南 5.2 末尾"按数据量选基座"） | **Instruct**（已做过指令微调） |
| 模板 | Alpaca | ChatML |
| 数据 | 1000 条 alpaca-cleaned | 厂商私有指令数据 |

两者比 = "**我的半成品 vs 别人家的成品**"，同时混入**基座不同 + 模板不同**两个变量。赢或输都**得不出"LoRA 有没有用"的结论**。

> 本地拿它粗看感受一下可以，但**别拿它下结论**。

---

## 二、正确设计：同一个基座，只切"挂不挂 adapter"

| 组 | 基座 | 提示词 | adapter |
|---|---|---|---|
| **对照组** | `unsloth/Qwen2.5-7B-bnb-4bit` | Alpaca 三行 | ❌ 不挂 |
| **实验组** | **同一次加载**的同一个模型 | 同一套（一字不差） | ✅ `load_adapter` 挂上 |

**唯一变量 = adapter**，而且必须在**同一进程、同一次加载**里切换——这样连显存布局与随机性都无关，比"跑两个 notebook 再比"干净得多。

---

## 三、跑法（Kaggle）

新建 Notebook（Accelerator 选 `GPU T4 ×2`，且 **首格必须是 `%env CUDA_VISIBLE_DEVICES=0`**，见指南 2.1），然后：

1. **贴指南 2.1 步骤 4 的安装格**（与训练时同一套，保证 Unsloth / transformers 版本一致）
2. **Add Input 挂上训好的 Output**，路径形如
   `/kaggle/input/<你的Output名>/qwen2.5-finetuned-final`
3. 跑下面的主对照代码

```python
import json, torch
from unsloth import FastLanguageModel

BASE    = "unsloth/Qwen2.5-7B-bnb-4bit"     # 必须与训练时完全一致
ADAPTER = "/kaggle/input/<你的Output名>/qwen2.5-finetuned-final"

model, tok = FastLanguageModel.from_pretrained(
    model_name=BASE, max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)


# 24 条题：英文同分布 8 + 中文探针 8 + 数学/知识探针 8
PROMPTS = [
    # ① 同分布英文指令（训练数据就是这类，看"贴不贴"）
    "Give three tips for staying healthy.",
    "Explain what a neural network is in simple terms.",
    "Write a thank-you email to a colleague who helped with a project.",
    "List five common mistakes beginners make when learning to code.",
    "Summarize the benefits of regular exercise in three sentences.",
    "What are the advantages of using version control in software development?",
    "Recommend three books for someone new to machine learning and explain why.",
    "Describe the water cycle in a way a 10-year-old could understand.",
    # ② 退化探针·中文（训练数据全英文，这里最可能出差异）
    "请用一句话解释什么是机器学习。",
    "把“今天天气很好”翻译成英文。",
    "介绍一下中国的四大发明。",
    "给我的年终总结写个开头。",
    "用三句话说明定期运动的好处。",
    "解释一下什么是过拟合。",
    "写一首关于秋天的五言诗。",
    "帮我给老板写一封请假邮件，理由是感冒。",
    # ③ 退化探针·数学 / 知识
    "If a shirt costs $25 after a 20% discount, what was the original price?",
    "What is 17 * 23? Show your steps.",
    "A train travels 240 km in 3 hours. What is its average speed in m/s?",
    "What is the capital of Australia?",
    "Who wrote \"Pride and Prejudice\"?",
    "Explain the difference between a list and a tuple in Python.",
    "Sort these numbers from smallest to largest: 3.2, 2.5, 3.15, 2.05.",
    "Convert the fraction 7/8 into a percentage.",
]


def alpaca(p):      # 两边同一套提示词，格式不能差一个字符
    return f"### Instruction:\n{p}\n\n### Response:\n"


@torch.no_grad()
def run(prompts):
    outs = []
    for p in prompts:
        ids = tok([alpaca(p)], return_tensors="pt").to("cuda")
        g = model.generate(**ids, max_new_tokens=256, do_sample=False)   # 贪心 → 可复现
        text = tok.batch_decode(g, skip_special_tokens=True)[0]
        outs.append(text.split("### Response:\n")[-1].strip())           # 只留回答
    return outs


base_out = run(PROMPTS)          # ① 对照组：不挂 adapter
model.load_adapter(ADAPTER)      # 同一进程内挂上，无需重新加载模型
ft_out   = run(PROMPTS)          # ② 实验组：挂了 adapter

json.dump({"prompts": PROMPTS, "base": base_out, "ft": ft_out},
          open("/kaggle/working/ab_raw.json", "w"), ensure_ascii=False, indent=2)
print("完成 → /kaggle/working/ab_raw.json")
```

> ⚠️ **两个细节别漏**：
> - **`do_sample=False`**（贪心解码）——采样会把随机性混进两组差异，让结论不可复现
> - **`max_new_tokens` 两边一致**——否则长度差会污染判断
>
> 若 `load_adapter` 报错，退路是重新 `from_pretrained` 后再挂——但**两组的解码参数必须完全相同**。

### 多轮探针（单独跑，因为要保留上文）

Alpaca 格式本身没有多轮结构，所以这里**最可能暴露格式崩坏**，值得单独测：

```python
MULTI_TURN = [
    ["What is LoRA?",
     "How does it save memory compared to full fine-tuning?"],
    ["介绍一下杭州这个城市。",
     "它有哪些值得去的景点？"],
]

@torch.no_grad()
def run_chat(turns):
    text = ""
    for t in turns:
        text += f"### Instruction:\n{t}\n\n### Response:\n"
        ids = tok([text], return_tensors="pt").to("cuda")
        g = model.generate(**ids, max_new_tokens=200, do_sample=False)
        out = tok.batch_decode(g, skip_special_tokens=True)[0]
        # 取最后一次 ### Response: 之后、直到它自己续写下一轮 Instruction 为止
        reply = out.split("### Response:\n")[-1].split("### Instruction:")[0].strip()
        text += reply + "\n\n"
    return text

# 两组各跑一遍（注意：挂/不挂的切换与上面同一进程内完成）
chat_base = [run_chat(t) for t in MULTI_TURN]
model.load_adapter(ADAPTER)
chat_ft   = [run_chat(t) for t in MULTI_TURN]
```

---

## 四、盲评（关键步骤）

**先别看输出**——否则你会不自觉地偏爱某一边。

1. **打乱 + 去标签**：每题把两组回答随机排成 A / B，抹掉模型名
2. **两轮、换位置**：第 2 轮把 A/B 对调 —— 消除"总偏好先出现那个"的位置偏差
3. **评分**：0-5 分，或用更省事的**成对偏好**（A 更好 / B 更好 / 平）
4. **客观指标**（直接算，不靠判断）：

| 指标 | 怎么算 |
|---|---|
| 平均输出长度 | `len(outs[i].split())` 取均值，两组对比 |
| 格式崩坏次数 | 回答里出现 `### Instruction:` 的次数 |
| 语言串台 | 中文题用英文回答、或反向的次数 |

> 最省事：把打乱后的对照表贴给 DeepSeek 当裁判 —— 就是 `llm-lora-qlora-finetuning-guide/微调评测方法A/eval_judge.py` 的思路。

---

## 五、结果表模板

**逐题表**

| 题号 | 类别 | 对照组(base) | 实验组(ft) | 偏好 | 备注 |
|---|---|---|---|---|---|
| 1 | 英文指令 | | | | |
| 9 | 中文 | | | | |
| 17 | 数学 | | | | |
| — | 多轮 | | | | |

**汇总表**

| 指标 | 对照组(base) | 实验组(ft) | 差异 |
|---|---|---|---|
| 英文指令 平均分 | | | |
| 中文 平均分 | | | |
| 数学/知识 平均分 | | | |
| 多轮 平均分 | | | |
| 平均输出长度（词） | | | |
| 格式崩坏次数 | | | |

---

## 六、怎么读结果（预期管理，最重要）

指南里写明：**base + Alpaca 路线需要"数千条"数据**；而本次只有 **1000 条**（3.2 的 `MAX_SAMPLES`）+ **2 轮** + r16，是很轻的改动。

> **合理预期：不会"变强"。最多看到风格更贴 Alpaca（更爱分点、更长）；中文 / 数学轻微退化，属正常且预期。**

| 观察到的 | 含义 | 下一步 |
|---|---|---|
| 英文更贴格式，通用能力基本不变 | 微调"生效但温和"，**流程验证成功** | 换中文数据 + 加量（阶段二下半场） |
| 通用能力明显退化（中文/数学变差） | **灾难性遗忘**苗头（数据单一 + 量少） | 降学习率 / 降轮次，或混入通用数据 |
| 两边几乎无差别 | 1000 条太轻，LoRA 未产生可测影响 | 加到数千条后重测 |

**哪种结果都不算失败**——它正是"要不要微调、要多少数据"的决策依据，也是 README 里那句"终局能力是决策力"的实践。

---

## 七、与「微调评测方法A」的关系

`llm-lora-qlora-finetuning-guide/微调评测方法A/` 那 4 个脚本（create_testset / eval_generate / eval_judge / eval_report）是现成的评测流水线，但属 **opt-125m 口径**，接 7B 要改 5 处：

| # | 改什么 | 从 → 到 |
|---|---|---|
| 1 | 模板 | `### Human/### Assistant` → Alpaca 三行 |
| 2 | 基座 | `facebook/opt-125m` → `unsloth/Qwen2.5-7B-bnb-4bit` |
| 3 | FT 端 | 加载 merged 目录 → 同基座 `load_adapter` |
| 4 | 评测集 | 纯英文 → 加中文退化探针 |
| 5 | 生成端 | 加 `do_sample=False` |

**本指南用轻量版**（一个 notebook 跑完 + 盲评）即可；要正式产出"可复现的评测报告"（阶段四），再按上表改造那套脚本。

---

## 八、产出物

- `ab_raw.json`：两组原始输出（**别只留截图**，截图无法复查）
- 第五节的**逐题表 + 汇总表**
- 一段结论：**哪类题变好了 / 哪类变差了 / 下一步动什么**
