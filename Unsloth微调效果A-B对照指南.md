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
    L = sum(len(x) for x in arr) / len(arr)     # ⚠️ 用字符数：中文无空格，split() 数的是"段数"
    print(f"{n}: 平均 {L:.0f} 字符 | 自续 '### Instruction:' {sum(x.count('### Instruction:') for x in arr)} 次")

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

> ⚠️ **长度一律读「字符数」**：中文没有空格，`len(x.split())` 数中文时数出来的是**段数**（一整段 170 字只算 `1 词`），跨中英文比长度会得出完全错误的结论。原因与实测见下面的陷阱小节。

### 分类读数（长度差是谁贡献的）

自检只看总数，这一格把三类拆开——**如果涨幅是中文题贡献的，那多半是"用英文绕圈"的退化信号，不是变好**：

```python
cats = ["英文"] * 8 + ["中文"] * 8 + ["数学/知识"] * 8      # 顺序同第三节 PROMPTS

for name, arr in [("base", B), ("ft", F)]:
    line = [f"{c} {sum(len(arr[i]) for i in range(24) if cats[i] == c) / 8:.0f} 字符"   # len() 不是 split()
            for c in ["英文", "中文", "数学/知识"]]
    print(f"{name:4s} " + " | ".join(line))


def ascii_heavy(s):        # 判断中文题是否用英文作答
    letters = [ch for ch in s if ch.isalpha()]
    return bool(letters) and sum(ch.isascii() for ch in letters) / len(letters) > 0.8


for name, arr in [("base", B), ("ft", F)]:
    cn = arr[8:16]         # 题 9–16 = 中文退化探针
    hit = [i + 9 for i, x in enumerate(cn) if ascii_heavy(x)]
    print(f"{name}: 中文题用英文作答 {len(hit)} / 8  命中题号 {hit}（#10 是翻译题，本身就该输出英文 → 见下）")
```

**怎么读**：

- 三类都均匀变长 → 通用的"Alpaca 风格"生效
- 只有其中一类猛涨 → 去看那几题的实际输出，大概率是绕圈 / 串台
- ⚠️ **某类 base 本来就近乎空（< 10 字符）→ 该类的 A/B 对照无意义**，见下

### ⚠️ 陷阱一：数中文长度**不能**用 `split()`

**这是本项目实测踩到的第一个坑**：中文没有空格，`len(x.split())` 数出来的**不是"词数"，而是"段数"**——一整段 170 字的回答只被算成 `1 词`：

| 题 | `len(x.split())` | `len(x)` | 汉字数 |
|---|---|---|---|
| [#11] 四大发明 | **1 词** | 170 字符 | 159 |
| [#14] 过拟合 | **1 词** | 160 字符 | 149 |

所以如果读数里出现「**base 中文平均 2 词**」，那**是指标造出来的假象，不是模型交白卷**——真实值是 **83 字符 / 72 汉字**，答得满满当当。

> **规则**：跨中英文比长度**一律用 `len()`（字符数）**；`split()` 只在纯英文语料上才有"词"的含义。

### ⚠️ 前置检查：base 某类是不是**真的**空

判空也用**字符数**。先跑这一格看清 base 到底输出了什么：

```python
# !r 会把空输出显示成 ''，把"话说一半被截断"暴露成少一个右引号
for i in range(8, 16):                       # 中文 8 题；要看英文块就把区间换成 range(0, 8)
    print(f"[{i+1}] {P[i]}\n  base: {B[i]!r}\n  ft  : {F[i]!r}\n")
```

**判读**：

- base 大量 `''` 或 `len() < 10` → **真的是基座不会**，该类只看 ft 的**绝对质量**（答对没、串台没），不跟 base 比
- base 输出其实很长、只是没被切出来 → 是缓存/切分问题，回 ③ 检查 `split("### Response:\n")[-1]`

#### 本项目实测（中文 8 题，2026-10-06，Kaggle T4 单卡）

**结论：base 没有交白卷 → 中文退化探针有效，能用来抓真信号。**

| 题 | base（字符） | ft（字符） | 偏好 |
|---|---|---|---|
| 9 一句话解释机器学习 | 30（答对） | 40（答对） | 平 |
| 10 翻译成英文 | 29（**直译，守约**） | 200（**改成论述**，悖约） | base |
| 11 四大发明 | 170 | 462（分点展开） | ft |
| 12 年终总结开头 | 146（写成"让我来写一下…"元叙述，**没给开头**） | 152（**给出正式开头**，守约） | ft |
| 13 三句话说运动好处 | 27（**只有 1 句**，悖约） | 89（**3 句**，守约） | ft |
| 14 解释过拟合 | 160 | 234 | 平 |
| 15 五言诗 | 51 | 51（后两句被改写） | 平 |
| 16 请假邮件 | 51（**话说一半断掉**） | 726（**整封英文**，串台） | base |
| **均值** | **83 字符** | **244 字符** | — |

三条可用的读法：

1. **长度**：三类全面变长，但**涨幅全压在两个探针类上**（同样的 `len()` 口径，全部 24 题）：

| 类别 | base | ft | 涨幅 |
|---|---|---|---|
| 英文（训练分布内） | 846 字符 | 1007 字符 | ×1.19 |
| 中文（探针） | 83 字符 | 244 字符 | **×2.94** |
| 数学/知识（探针） | 318 字符 | 524 字符 | **×1.65** |

> 训练分布内的英文只长了 19%，两个探针类却长了 65% / 194% → **变长主要发生在"本来不该按 Alpaca 铺长回答"的地方**，这是**风格外溢**，不是能力提升。
2. **约束遵守是双向的**：ft 修好了「三句话」「年终总结开头」，却弄坏了「翻译」——**不能只报喜**
3. **退化信号 = 语言串台**：`[16]` 中文题被答成整封英文邮件，这是本次最硬的一个退化证据

> ⚠️ **`ascii_heavy()` 会把第 10 题算成串台**——那题本身要求输出英文（base、ft 都是英文，属正常）。**扣掉它再读**：base **0/8**、ft **1/8（只有 `[16]`）**。所以原始读数里的「base 1 / 8」是**纯误报**。
> 本次实跑已把命中题号打出来佐证：base `[10]`、ft `[10, 16]` —— 与上述判断一致，**base 只有误报、ft 多出一题真串台**。


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
| 平均输出长度 | `len(outs[i])` 取均值（**字符**，中英通用；**别用 `split()`**），两组对比 |
| 格式崩坏次数 | 回答里出现 `### Instruction:` 的次数 |
| 语言串台 | 中文题用英文回答、或反向的次数（**排除 #10 翻译题**，它本就该输出英文） |

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

把打印出来的内容**整段贴给裁判模型**（DeepSeek 即可）。
`ab_blind.json` 里的 `A_is_ft` 是**答案键**——评分完成后再用它解盲、把分数归到 base / ft 两边。

### 裁判提示词（连同上一格输出一起发给 DeepSeek）

rubric 与你的 `llm-lora-qlora-finetuning-guide/微调评测方法A/eval_judge.py` 保持一致，两处口径不会打架：

```
你是严格的评测员。下面有 N 道题，每题两个匿名回答（A / B）。只输出 JSON，不要任何解释。

评分 rubric（0-5）：
5 = 完全正确且满足指令的每一条硬约束（数量、一句话、字数、指定开头等）
4 = 正确，基本满足约束，有小瑕疵
3 = 方向对，部分正确或只满足部分约束
2 = 有相关内容但明显错误，或忽略了主要约束
1 = 严重跑题或近乎空白
0 = 空、乱码、无关

评分只看回答本身，不要因为篇幅长就给高分（啰嗦 ≠ 好）。

整体输出：{"items": [{"id": <题号>, "score_A": <0-5>, "score_B": <0-5>, "winner": "A"|"B"|"平", "comment": "<20 字内理由>"}, ...]}
```

**第 2 轮换位置**（消位置偏差）——把每题的 A/B 对调后再判一遍，**另存一份别覆盖**：

```python
import json
b = json.load(open("/kaggle/working/ab_blind.json"))
for r in b:
    r["A"], r["B"] = r["B"], r["A"]
json.dump(b, open("/kaggle/working/ab_blind_swapped.json", "w"), ensure_ascii=False, indent=2)
for r in b:
    print(f"### #{r['id']} {r['prompt']}\n[A]\n{r['A']}\n[B]\n{r['B']}\n")
```

> 两轮结论**不一致的题**单独看——那说明偏好不稳定，别硬下结论。

### 解盲 + 汇总（裁判结果存盘后跑这一格）

把裁判返回的 `{"items": [...]}` 存成 `/kaggle/working/ab_verdict.json`，这格**自动解盲并算出第五节的汇总表**：

```python
import json, statistics

blind = {r["id"]: r for r in json.load(open("/kaggle/working/ab_blind.json"))}
V = json.load(open("/kaggle/working/ab_verdict.json"))["items"]
cats = ["英文"] * 8 + ["中文"] * 8 + ["数学/知识"] * 8      # 顺序同第三节 PROMPTS

rows = []
for v in V:
    ft_is_a = blind[v["id"]]["A_is_ft"]                 # 答案键：ft 是不是被放成了 A
    base_s = v["score_B"] if ft_is_a else v["score_A"]
    ft_s   = v["score_A"] if ft_is_a else v["score_B"]
    w = str(v.get("winner", "平")).strip()
    win = "tie" if w in ("平", "tie", "") else \
          ("ft" if (w.upper().startswith("A") == ft_is_a) else "base")
    rows.append({"id": v["id"], "cat": cats[v["id"] - 1], "base": base_s, "ft": ft_s, "win": win})

print(f"{'类别':<10}{'base均分':>9}{'ft均分':>9}{'ft胜':>6}{'base胜':>8}{'平':>5}")
for c in ["英文", "中文", "数学/知识"]:
    g = [r for r in rows if r["cat"] == c]
    print(f"{c:<10}{statistics.mean(r['base'] for r in g):>9.2f}"
          f"{statistics.mean(r['ft'] for r in g):>9.2f}"
          f"{sum(r['win'] == 'ft' for r in g):>6}"
          f"{sum(r['win'] == 'base' for r in g):>8}"
          f"{sum(r['win'] == 'tie' for r in g):>5}")

d = json.load(open("/kaggle/working/ab_raw.json"))
print("\n平均输出长度: base %.0f 字符 | ft %.0f 字符" % (
    sum(len(x) for x in d["base"]) / len(d["base"]),
    sum(len(x) for x in d["ft"]) / len(d["ft"])))
```

> **判据是相对差，不是绝对分**：ft 均分在**英文**类上 ≥ base、且在**中文/数学**类上掉幅 < 0.5 分，才算"生效且没伤到通用能力"。掉幅 ≥ 1 分 → 按第六节当"灾难性遗忘苗头"处理。

---

## 五、结果表模板

**逐题表**

| 题号 | 类别 | 对照组(base) | 实验组(ft) | 偏好 | 备注 |
|---|---|---|---|---|---|
| 1 | 英文指令 | | | | |
| 9 | 中文 | | | | |
| 17 | 数学 | | | | |
| — | 多轮 | | | | |

**汇总表**（前三行要跑完盲评才有；后三行是**客观指标，已实测可直接抄**）

| 指标 | 对照组(base) | 实验组(ft) | 差异 |
|---|---|---|---|
| 英文指令 平均分 | | | |
| 中文 平均分 | | | |
| 数学/知识 平均分 | | | |
| 多轮 平均分 | | | |
| 平均输出长度（字符，全 24 题） | ≈ 416 | ≈ 592 | **×1.42** |
| 格式崩坏次数（自续 `### Instruction:`） | 0 | 0 | 0 |
| 语言串台次数（扣掉 #10 翻译题） | 0 | **1**（`[16]`） | +1 |

> 长度按类拆开看：英文 846 → 1007（×1.19）、中文 83 → 244（×2.94）、数学/知识 318 → 524（×1.65）。
> 中文 8 题的逐题长度与偏好已列在第三节实测表——**那张表就是逐题表的"中文行"**，盲评出分后把分数补到本表的逐题格里。

---

## 六、怎么读结果（预期管理，最重要）

指南里写明：**base + Alpaca 路线需要"数千条"数据**；而本次只有 **1000 条**（3.2 的 `MAX_SAMPLES`）+ **2 轮** + r16，是很轻的改动。

> **合理预期：不会"变强"。最多看到风格更贴 Alpaca（更爱分点、更长）；中文 / 数学轻微退化，属正常且预期。**

### ⚠️ 先认清这套对照能回答什么（结论上限）

对照组是 **Base**（不是 Instruct），所以它**给不出"日常可用的基线"**。注意这**不等于**"base 不会答题"——本项目实测（中文 8 题）base **题题都有实质内容**（均值 83 字符），只是：

- **约束遵守更松**（#13 只要 1 句、#12 不给开头、#16 话说一半断掉）
- **英文块倾向于"续写"而非"作答"**（base 单题 ~846 字符 ≈ 135 词，量不小；要确认是"作答"还是"续写下一题"，dump `range(0, 8)`）

| 你想问的问题 | 该拿谁当对照 | 本指南能答吗 |
|---|---|---|
| **adapter 有没有生效？** | 同基座：挂 vs 不挂 | ✅ **能**——这正是本指南的目标 |
| **微调后在中文 / 数学上退步了吗？** | 同基座（base 有内容就算数） | ✅ **能**——本次正是靠它抓到 `[16]` 的整封英文串台 |
| 微调后值不值得用？ | 你平时真在用的 **Instruct** 版 | ❌ 不能（另一个实验） |
| 数据量 / 配比的影响？ | 两个 adapter 互比 | ❌ 不能（阶段二下半场做） |

**本指南的结论止于"adapter 生效 + 有没有伤到中文/数学"**：`0/24 逐字相同` + 三类长度全面上升，说明模型确实学会了 Alpaca 指令格式——**流程验证成功**，但**不等于微调质量好**。

> 顺带修正第一节的适用范围：那句"别拿 `qwen2.5:7b` 当对照"成立的前提，是你想答**上表第一个问题**。如果哪天你要问的是"**微调后比我平时用的 Instruct 版好还是坏**"，那 Instruct 就恰恰是**正确的对照**——只是它回答的是另一个问题（真实使用基线）。

| 观察到的 | 含义 | 下一步 |
|---|---|---|
| 三类都变长、格式稳定、0/24 逐字相同 | adapter **生效**，流程验证成功 | 换中文数据 + 加量（阶段二下半场） |
| base 某类 `len()` **< 10 字符**（真·空）而 ft 明显更长 | 该类 A/B **对照失效**，测不出退化 | 该类只看 ft 绝对值（答对没？串台没？） |
| ft 的**绝对表现**差（中文题答英文、数学答错） | **灾难性遗忘 / 欠拟合**苗头 | 换中文数据、加量，或降学习率 |
| 两边几乎无差别 | 1000 条太轻，LoRA 未产生可测影响 | 加到数千条后重测 |

**哪种结果都不算失败**——它正是"要不要微调、要多少数据"的决策依据，也是 README 里那句"终局能力是决策力"的实践。

#### 本次实测的结论（可直接引用）

- **adapter 生效**：`0/24 逐字相同`、0 空输出、自续 `### Instruction:` 0 次
- **风格变化**：三类全面变长（字符）——英文 **846 → 1007**（×1.19）、中文 **83 → 244**（×2.94）、数学/知识 **318 → 524**（×1.65）；**涨幅全压在两个探针类上** → 是 Alpaca 长回答风格**外溢**，不是能力提升
- **已出现退化信号**：`[16]`（中文写请假邮件）ft 输出**整封英文**；`[10]`（翻译题）ft 把"直译"改成"论述"——命中题号 base `[10]`、ft `[10, 16]`，**扣掉 #10 误报后串台 base 0/8、ft 1/8**
- **一句话**：**Alpaca 风格学到了，中文/数学开始被英文和啰嗦侵蚀**（与第六节开头"1000 条 + 2 轮不该期待变强"的预期一致）→ 下一步换中文数据 + 加量


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
- `ab_blind_swapped.json`：第 2 轮换位置后的对照（与第 1 轮一起判"偏好稳不稳"）
- `ab_verdict.json`：裁判返回的 `{"items": [...]}`（**原始评分记录**，解盲前不要手改）
- 第五节的**逐题表 + 汇总表**（汇总表可由第四节的解盲格直接算出）
- 一段结论：**哪类题变好了 / 哪类变差了 / 下一步动什么**

---

## 附：安装格的唯一性约定

第三节 ① 的安装格与《Unsloth微调实战指南.md》2.1 步骤 4、《Unsloth与原生HuggingFace对照.md》4.2 的安装格**逐字相同**——共 **3 份拷贝**。它们代表同一个被钉死的版本组合（`transformers==4.56.2` / `trl==0.22.2` 等），**改一处必须同改三处**，否则会出现"训练用 A 版本、对照跑 B 版本"这类隐蔽的口径错位。
