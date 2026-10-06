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

新建 Notebook（Accelerator 选 `GPU T4 ×2`），然后三步。

> ⚠️ **首格必须是 `%env CUDA_VISIBLE_DEVICES=0`**——Kaggle 只给 `GPU T4 ×2`、**没有单卡档位**，不限的话 Unsloth 会把模型切开跨两张卡跑（张量并行）：**不加速、还白吃一份时长额度**（实测双卡 300 步与单卡单步耗时基本持平，跨卡通信把第二张卡的收益吃光了）。
> **时序是硬要求**：这一行必须**早于任何 `import torch` / 任何 CUDA 调用**，晚一步就无效、只能重启会话。判据：加载模型时横幅 `Num GPUs = 1`，且**不出现** `output head -> cuda:1`。

**① 装环境** —— 贴下面这格（**与训练时同一套**，保证 Unsloth / transformers 版本一致）：

```python
import re
import torch

v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")

!pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
!pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
!pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
```

> 与指南 2.1 步骤 4 的安装格**逐字一致**（三份文档共用同一格；改动时要同步，见文末）。

**② 挂 adapter** —— **Add Input 挂上训好的 Output**，路径形如
`/kaggle/input/<你的Output名>/qwen2.5-finetuned-final`

**③ 跑对照（一次跑完单轮 + 多轮）** —— 贴下面这段。**顺序是关键**：base 组（单轮 + 多轮）全部排在 `load_adapter` 之前，之后只挂一次、不再需要"关"：

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


# 多轮探针：Alpaca 没有多轮结构，这里最容易暴露格式崩坏
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


# ---- base 组：全部跑在挂 adapter 之前（唯一变量才是 adapter）----
base_out  = run(PROMPTS)                          # 单轮 base
chat_base = [run_chat(t) for t in MULTI_TURN]     # 多轮 base

model.load_adapter(ADAPTER)                       # 只挂这一次，之后再没有"关"的需求

# ---- ft 组 ----
ft_out  = run(PROMPTS)                            # 单轮 ft
chat_ft = [run_chat(t) for t in MULTI_TURN]       # 多轮 ft

json.dump({"prompts": PROMPTS, "base": base_out, "ft": ft_out,
           "multi_base": chat_base, "multi_ft": chat_ft},
          open("/kaggle/working/ab_raw.json", "w"), ensure_ascii=False, indent=2)
print("完成 → /kaggle/working/ab_raw.json")
```

> ⚠️ **三个细节别漏**：
> - **顺序**：`base` 组必须排在 `model.load_adapter()` **之前**——Unsloth 这条路径返回的是模型本体（`Qwen2ForCausalLM`）、**没有 `disable_adapter()`**，adapter 一旦挂上，在进程内就没有"临时关掉"的开关（官方 issue #2688 / #36 同款现象）。这是把多轮并进本格的原因。
> - **`do_sample=False`**（贪心解码）——采样会把随机性混进两组差异，让结论不可复现。
> - **`max_new_tokens` 两边一致**——否则长度差会污染判断。

### 自检（先确认这次跑得有意义）

**最怕两组输出一模一样**——那说明 adapter 没真正生效，后面的盲评等于在比两个相同的模型。跑完 ③ 先跑这一格：

```python
import json
d = json.load(open("/kaggle/working/ab_raw.json"))
P, B, F = d["prompts"], d["base"], d["ft"]
print("单轮题数", len(P), "| base", len(B), "| ft", len(F))
print("单轮·两组逐字相同的题数:", sum(1 for b, f in zip(B, F) if b == f), "/", len(P))
print("空输出 base", sum(1 for x in B if not x.strip()), "| ft", sum(1 for x in F if not x.strip()))
for n, arr in [("base", B), ("ft", F)]:
    L = sum(len(x.split()) for x in arr) / len(arr)
    print(f"{n}: 平均 {L:.0f} 词 | 自续 '### Instruction:' {sum(x.count('### Instruction:') for x in arr)} 次")

# 多轮：③ 合并前的旧 json 只有 3 个键，缺失时跳过（单轮结论不受影响）
if "multi_base" in d and "multi_ft" in d:
    MB, MF = d["multi_base"], d["multi_ft"]
    print("多轮", len(MB), "/", len(MF),
          "| 逐字相同", sum(1 for a, b in zip(MB, MF) if a == b), "/", len(MB),
          "| 自续 '### Instruction:'",
          sum(x.count("### Instruction:") for x in MB), "/",
          sum(x.count("### Instruction:") for x in MF))
else:
    print("多轮：本 json 无 multi_base / multi_ft（旧版产出，③ 合并前跑的）")
    print("      → 单轮结论不受影响；要补多轮，重跑 ③ 整块即可（base 组会自动前置）")
```

**判据**：单轮题数应为 `24 / 24 / 24`、多轮 `2 / 2`；**「逐字相同」应远小于 24**（通常能差一半以上）；空输出为 `0`。
若「逐字相同 = 24」→ 回第三节检查 `load_adapter` 是否真的执行了（adapter 未生效时两组必然一致）。

### 为什么测多轮（代码已并入 ③）

Alpaca 格式本身**没有多轮结构**，所以这里**最可能暴露格式崩坏**——模型可能忘记上一轮、或干脆自己续写下一轮 `### Instruction:`。2 组多轮对话（各 2 轮）的原始输出在 `ab_raw.json` 的 **`multi_base` / `multi_ft`**（采集代码见 ③ 的 `run_chat`）。

**判据**：`### Instruction:` 出现次数应为 **0**（自续即格式崩）；再对比两组是否丢上下文、答非所问。

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

### 盲评对照怎么生成（直接粘）

固定随机种子 → 可复现；**答案键存在 json 里**，评完再解盲：

```python
import json, random
d = json.load(open("/kaggle/working/ab_raw.json"))
P, B, F = d["prompts"], d["base"], d["ft"]
random.seed(0)                                  # 固定种子 → 可复现
blind = []
for i, (p, b, f) in enumerate(zip(P, B, F), 1):
    a_is_ft = random.random() < 0.5             # 随机决定 ft 放到 A 还是 B
    A, Bs = (f, b) if a_is_ft else (b, f)
    blind.append({"id": i, "prompt": p, "A": A, "B": Bs, "A_is_ft": a_is_ft})
json.dump(blind, open("/kaggle/working/ab_blind.json", "w"), ensure_ascii=False, indent=2)

for r in blind:
    print(f"### #{r['id']} {r['prompt']}\n[A]\n{r['A']}\n[B]\n{r['B']}\n")
print("（第 2 轮：把每题 [A]/[B] 对调再判一遍，消除位置偏差）")
```

把打印出来的内容**整段贴给裁判模型**（DeepSeek 即可），要求逐题输出 `A更好 / B更好 / 平` + 0-5 分。
`ab_blind.json` 里的 `A_is_ft` 是**答案键**——评分完成后再用它解盲、把分数归到 base / ft 两边。

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

- `ab_raw.json`：全部原始输出——单轮 `base` / `ft` + 多轮 `multi_base` / `multi_ft`（**别只留截图**，截图无法复查）
- `ab_blind.json`：单轮盲评对照（含 `A_is_ft` 答案键），便于复查评分过程
- 第五节的**逐题表 + 汇总表**
- 一段结论：**哪类题变好了 / 哪类变差了 / 下一步动什么**

---

## 附：安装格的唯一性约定

第三节 ① 的安装格与《Unsloth微调实战指南.md》2.1 步骤 4、《Unsloth与原生HuggingFace对照.md》4.2 的安装格**逐字相同**——共 **3 份拷贝**。它们代表同一个被钉死的版本组合（`transformers==4.56.2` / `trl==0.22.2` 等），**改一处必须同改三处**，否则会出现"训练用 A 版本、对照跑 B 版本"这类隐蔽的口径错位。
