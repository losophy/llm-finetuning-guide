# 微调评测 · 方法 A 实操指南：自建评测集 + LLM 打分（opt-125m）

> 依据《简历量化数据补测指南》"一、微调评测 · 方法 A"展开。目的：给 LLM 微调（LoRA/QLoRA）项目补上**真实、可复现、可被追问**的量化数字——"微调后比 base 强多少"。
>
> ⚠️ 原则与补测指南一致：**所有数字必须实测后回填，严禁编造**。本文所有示例数字一律为占位符（`X`/`→`），跑完脚本替换。
>
> 本环境使用 opt-125m（流程验证/教学模型，125M 参数），因此补测指南里的**方法 B（公开基准 MMLU/GSM8K）不适用**，本文只写方法 A。

---

## 目录

1. [方法 A 是什么、产出什么指标](#1-方法-a-是什么产出什么指标)
2. [整体流程与目录结构](#2-整体流程与目录结构)
3. [Step 1 构建评测集](#3-step-1-构建评测集)
4. [Step 2 双模型批量推理（base vs 微调后）](#4-step-2-双模型批量推理base-vs-微调后)
5. [Step 3 LLM-as-judge 打分](#5-step-3-llm-as-judge-打分)
6. [Step 4 指标计算与报告](#6-step-4-指标计算与报告)
7. [可信度与面试追问准备](#7-可信度与面试追问准备)
8. [简历回填写法](#8-简历回填写法)
9. [附：命令速查与常见问题](#9-附命令速查与常见问题)

---

## 1. 方法 A 是什么、产出什么指标

**测什么**：微调后模型在你关心的"目标场景"上的表现，**和 base 模型（微调前）做同题对比**。因为微调本身不是目的，目的是证明"微调带来了可感知的提升"。

**为什么用 LLM 打分而不是人工**：50-100 条逐条人工评太慢；强模型（DeepSeek/GPT）做裁判，给 0-5 分，速度与一致性都够，且面试时口径统一、可复现。

**最终落到简历上只需两个数**（与补测指南保持一致）：

| 指标 | 定义 | 怎么算 |
|---|---|---|
| **指令遵循率** | 输出"完全符合指令硬性要求"的比例（格式/条数/句式/长度等约束是否照做） | 遵循条数 ÷ 总条数（judge 逐条判 true/false） |
| **平均分提升** | LLM 裁判 0-5 分，微调后平均分 − base 平均分 | judge 打分后算均值差，可再算相对提升 % |

**本环境的现实约束（opt-125m）**：

- opt-125m 是 125M 的小模型，**中文能力接近零**，而训练数据 guanaco 是英文对话 → **评测集建议用英文**，与训练分布同语言，分数才真实、有区分度。
- 小模型"踮脚够得着"的任务才有对比意义：概念解释、简短指令、简单格式约束（一句/三点/Yes-No）。**不要上** JSON schema、长文写作、数学推理这类连 base 都完全不会的任务——两边都拿 0 分，测不出提升。
- 分数绝对值可能偏低（base 常落在 1-3 分区间），**这不丢人**：简历要的是"提升多少"，不是"考了多少分"。如实呈现即可。

---

## 2. 整体流程与目录结构

```
llm-lora-qlora-finetuning-guide/
├─ opt-125m-merged/                       ← 微调合并模型（脚本自动定位到上级目录）
├─ lora_finetune_opt.py 等训练脚本
└─ 微调评测方法A/                          ← 本指南全部文件所在文件夹
   ├─ 微调评测方法A_自建评测集与LLM打分指南.md
   ├─ create_testset.py                   ← ① 建评测集 → evalset/test.jsonl（当前 60 条）
   ├─ eval_generate.py                    ← ② 批量推理 base vs 微调模型
   │                                        ├─ outputs/base.jsonl
   │                                        └─ outputs/ft.jsonl
   ├─ eval_judge.py                       ← ③ LLM 盲评打分 → outputs/scores.json
   ├─ .env                                (API 密钥: LLM_* 或 DEEPSEEK_API_KEY, ③ 自动读取)
   ├─ eval_report.py                      ← ④ 出报告 → outputs/eval_report.md
   ├─ evalset/test.jsonl                  （脚本生成，指令 + reference）
   └─ outputs/                            （脚本生成，jsonl / 报告留档）
```

**路径约定**：四个脚本都以"脚本所在文件夹（`微调评测方法A/`）"为基准生成 `evalset/` 与 `outputs/`；微调合并模型则自动定位到**上级目录** `llm-lora-qlora-finetuning-guide/opt-125m-merged`。因此无论你在哪个目录执行 `python 微调评测方法A/xxx.py`，路径都不会错。

| 资产 | 定位方式 | 说明 |
|---|---|---|
| base 模型 | `facebook/opt-125m`（固定） | HuggingFace 下载（首次约 250MB，走 hf-mirror） |
| 微调合并模型 | 脚本自动取上级目录的 `opt-125m-merged` | 若模型换位置，改脚本里 `FT_MODEL` 即可 |
| prompt 模板 | `### Human: {question}\n### Assistant: ` | 与 `lora_finetune_opt.py` / `test_merged_model.py` 完全一致 |

> 运行解释器：与你跑训练/推理一致的 Python 环境（你机器上是 `C:/Users/Admin/AppData/Local/Programs/Python/Python310/python.exe`），依赖 torch / transformers / peft / requests 与训练时相同。

---

## 3. Step 1：构建评测集

### 3.1 要求与原则

1. **数量**：50-100 条。本指南按 **60 条** 设计（足够算平均分与分项，opt-125m 推理也快）。
2. **结构**：每条 = `instruction`（指令，发给模型的 prompt）+ `reference`（参考答案，给 judge 对照，不用来训练）。
3. **同分布但不同源**：题型、语言、难度贴近你的训练数据（guanaco 风格英文指令问答），但**具体题目必须是你新写的**，不要直接抄训练集里的样本——否则评测 = 背题，面试一追问就穿帮。
4. **难度匹配 opt-125m**：见第 1 节约束。
5. **类别要均衡**：让"指令遵循率"有得测（放一批带硬约束的题），让"平均分"有得比（放一批知识/解释题）。

### 3.2 类别与题量分配建议（60 条）

| 类别 | 题量 | 作用 | 示例题型 |
|---|---|---|---|
| A. 概念解释 | 15 | 测知识回答质量（拉平均分） | What is / How does / Explain ... |
| B. 指令遵循·格式约束 | 15 | 测"照做能力"（主要喂遵循率） | 列三点 / 一句话 / Yes-No / 以 XX 开头 |
| C. 判断比较 | 10 | 测理解深度 | A 与 B 区别 / 是否 / 为什么 |
| D. 改写润色 | 10 | 测语言控制 | 改正式 / 缩短 / 改写为问句 |
| E. 简短建议生成 | 10 | 测应用能力（带小约束） | 给一条建议并说明 |

> 想让简历项目看起来更"业务向"，可把 A/E 类换成你真正的目标场景（如"把日志改写成 XX 风格"），题型结构不变，只换题干。

### 3.3 评测集文件 `evalset/test.jsonl`（当前 60 条）

用文件夹内的 `create_testset.py` 生成（产物在脚本所在目录的 `evalset/test.jsonl`）。**`TEST_SET` 已按 3.2 表的分配内置满 60 条**（A 15 / B 15 / C 10 / D 10 / E 10）。（起步阶段曾先用 24 条试跑验证全流程，跑通后才补齐到 60——先小批验证、再放量的节奏便于调试与校准题目难度。）

> **引用脚本 `create_testset.py`**（与本文同目录）
> - 做什么：内置 60 条示例题（5 类分布 A 15 / B 15 / C 10 / D 10 / E 10，对应 3.2 表题型）→ 逐条写成 `{"id","category","instruction","reference"}` 的 `evalset/test.jsonl`
> - 运行：`python create_testset.py`（当前 60 条）；自定义输出 `python create_testset.py --out 路径.jsonl`
> - 完整题目清单见生成的 `evalset/test.jsonl`；**换题/扩量 = 编辑脚本内 `TEST_SET` 列表**，保存后重跑即可覆盖


**若要换题 / 扩到 100+ 条**（如把 A/E 类换成业务向题目、或扩量增加统计稳定性）——编辑 `TEST_SET` 后重跑 ① 即可，注意：
- 同一概念别问 N 遍相似问法，换知识点；
- B 类硬约束要**明确、可机械判定**（"三点""Yes/No""少于 15 词""以 XX 开头"），judge 才好判"遵循了没有"；
- 完成后自查一遍：**确保任意一条都不等于训练数据里的原句**。

---

## 4. Step 2：双模型批量推理（base vs 微调后）

### 4.1 为什么必须严格控制变量

比较才有意义的前提是：**除了"是否微调"这一个变量，其余全部一致**——同一批题、同一个 prompt 模板、同一套解码参数、同一个随机种子。任何一项不一致，分数差就说不清来源，面试追问即破功。

### 4.2 生成脚本 `eval_generate.py`

> **引用脚本 `eval_generate.py`**（与本文同目录），双模型同题批量推理，变量控制已写死在脚本里：
> - 同一份 `evalset/test.jsonl` + 同一 prompt 模板 `### Human: {instruction}\n### Assistant: `（与训练/自测脚本一致）
> - 同一解码参数（`temperature=0.5`、`top_p=0.9`、`top_k=50`）与固定随机种子 42，保证可复现
> - base 固定 `facebook/opt-125m`（首次自动下载，走 hf-mirror）；微调合并模型自动定位到上级目录 `opt-125m-merged`（模型换位置则改脚本内 `FT_MODEL`）
> - 产物：`outputs/base.jsonl`、`outputs/ft.jsonl`（与 `test.jsonl` 的 `id` 一一对应），终端抽样打印前 3 条供肉眼检查
>
> 运行：`python eval_generate.py`


**跑前自查**：
- 先肉眼抽查 base / ft 各 5-10 条：opt-125m 有时会输出空串或重复字符。若某类大面积空输出，考虑该题难度过高（对 125m 不现实），应换题而不是硬留——评测题必须"模型真能答出东西"才有区分度。
- 若发现 base 就很好、微调后反而变差，先别慌：检查训练轮数是否过拟合（guanaco 9k+ 条只跑 1 epoch 一般不会），以及评测题是否离训练分布太远。详见第 7 节。

---

## 5. Step 3：LLM-as-judge 打分

### 5.1 为什么必须"盲评"

直接把答案和模型名一起丢给裁判，裁判会有先入为主的偏好。正确做法：**把 base 和 ft 的答案混在一起、打乱顺序、不告诉裁判谁是谁**，最后再按 id 归位。这样分数差的唯一解释就是答案本身的质量差。

### 5.2 打分口径（写死在 prompt 里，保证可复现）

- **分数 0-5**（整数）：
  - 5 = 内容正确且完全满足指令约束（或与参考等价）
  - 4 = 正确，基本满足约束，有小瑕疵
  - 3 = 方向对，部分正确或只满足部分约束
  - 2 = 有相关内容，但明显错误或忽略主要约束
  - 1 = 严重偏离 / 只有只言片语
  - 0 = 空回答 / 乱码 / 与问题无关
- **follows_instruction（true/false）**：只看**硬性约束**是否照做（三点、一句话、Yes/No、词数、以 XX 开头等）；题目没有硬约束时一律判 true。**内容对错不计入此布尔值**（对错已体现在分数里）。

### 5.3 打分脚本 `eval_judge.py`（OpenAI 兼容，支持 DeepSeek / 百炼 qwen 等）

> **引用脚本 `eval_judge.py`**（与本文同目录），LLM-as-judge 盲评打分：
> - 读取 `outputs/base.jsonl` + `outputs/ft.jsonl`；5.2 的 rubric 已写死在脚本 `JUDGE_SYSTEM` 中随 prompt 发给裁判
> - **盲评**：base/ft 全部答案打乱顺序（固定洗牌种子）逐条送裁判，裁判不知道谁是谁；返回 0-5 分数 + `follows_instruction` + 裁判 `comment`（存盘供审计）
> - **空答案直接记 0 分、不送 API**：模型未产出内容即视为 0 分（rubric 中 empty=0）。实测若把空串直接发给裁判，部分裁判会误评高分（曾把空答案评为 5 分），故脚本对空答案短路处理，不消耗调用
> - 单条调用失败自动重试 3 次，仍失败记为 `-1`（出报告时剔除）
> - 产物：`outputs/scores.json`（meta 记录实际使用的 model / base_url，留作复现口径）
>
> 运行（密钥放同目录 `.env`，脚本自动读取，无需每次设环境变量）：
> 1. 在 `.env` 填好密钥——认 `LLM_API_KEY`/`LLM_MODEL_NAME`/`LLM_BASE_URL`（OpenAI 兼容网关通用）或 `DEEPSEEK_API_KEY`（DeepSeek 官方）；本文件夹已带一份可直接用的 `.env`
> 2. `python eval_judge.py`（`--model` / `--base-url` 显式传参会覆盖 .env；无 .env 时默认 DeepSeek：`deepseek-chat` + `api.deepseek.com/v1`）
> 3. 成本约 2×题数 次调用（60 题约 120 次），通常不足 1 元


**跑完后**：`scores.json` 里有 `-1` 的记录（调用失败）要在算指标时剔除；若 -1 超过 5 条，建议降速重跑这几条。

---

## 6. Step 4：指标计算与报告

### 6.1 指标定义（写清楚，面试口径一致）

| 指标 | 公式 |
|---|---|
| base / ft 平均分 | Σscore ÷ 有效条数 |
| 平均分提升（绝对值） | ft 平均分 − base 平均分 |
| 平均分提升（相对） | (ft − base) ÷ base × 100% |
| 指令遵循率 base / ft | follows=true 条数 ÷ 有效条数（**分模型各自统计**） |
| 胜出率（win rate） | ft_score > base_score 的条数 ÷ 总条数（并列不计） |
| 分项得分 | 按 category 分别算上述平均分 |

### 6.2 报告脚本 `eval_report.py`

> **引用脚本 `eval_report.py`**（与本文同目录）：
> - 读取 `outputs/scores.json` + `evalset/test.jsonl`，按 6.1 公式计算总指标与分项指标（自动剔除 `-1`）
> - 产物：`outputs/eval_report.md`（总表 + 按 category 分项表 + 复现证据说明，终端同步打印）
>
> 运行：`python eval_report.py`


**看报告时的判断指引**：
- 期望形态：ft 平均分与遵循率整体高于 base，胜出率 > 60%；概念/指令类提升明显，改写类可能提升有限。
- 若**整体几乎无提升甚至下降**：优先怀疑评测分布 ≠ 训练分布，或训练不充分。对策与解释口径见第 7 节——**如实呈现原因比硬编一个好看的数字有用得多**。

---

## 7. 可信度与面试追问准备

数字要能扛住面试官三个方向的追问：

**Q1：这些题是不是背过的训练题？**
- 答：评测集 60 条全部为评测阶段新写，未出现在训练集；题型/语言/难度刻意贴近训练分布以测"迁移到同类任务"的能力，但题干均不同源。可当场打开 `evalset/test.jsonl` 与训练脚本对照。

**Q2：分数是谁打的？怎么保证公平？**
- 答：强 LLM 作裁判（本流程经 `.env` 配置的 OpenAI 兼容接口调用，如百炼 qwen3-max-preview；换 DeepSeek 只需改 .env/传参），0-5 固定 rubric（可复述，写死在 `eval_judge.py` 的 JUDGE_SYSTEM）；base 与 ft 答案**打乱盲评**，裁判不知道谁是谁；生成阶段两模型共用同一 prompt 模板、同解码参数、固定随机种子。产物 `outputs/base.jsonl` / `outputs/ft.jsonl` / `scores.json`（meta 记录了实际 judge 模型）全程留档（均在 `微调评测方法A/outputs/`），可复跑复现。

**Q3：为什么用 opt-125m？这个分数能说明什么？**
- 口径：opt-125m 用于**端到端验证 LoRA/QLoRA 流程与评测方法论**（125M 参数，训练分钟级，迭代快）；评测分数体现的是"同题对比下微调带来的相对提升"，绝对值受模型容量限制。若要更强的说服力，同样的评测集与脚本可直接套到 Qwen2-7B/更大模型（仓库 `qwen2-7b-lora` 目录即 7B 链路）。切勿把 125m 的分数包装成 7B 级能力。

**防翻车清单**：
1. 所有数字从 `eval_report.md` 抄，跑完才填，任何一步没跑完就不写。
2. 报告头部日期/设备/judge 模型补全——面试要细节时答得上。
3. 若提升不理想：先检查评测题难度与分布，再检查训练（epoch、数据量），把"为什么"想清楚；简历可改为只写成立的那部分（如仅指令遵循率提升明显，就只报遵循率）。
4. 别把"遵循率 96%"这类 AI 编造的示例数字当目标——opt-125m 实测很可能到不了 96%。**真实的小提升 > 漂亮的大数字**。

---

## 8. 简历回填写法

实测后，把第 6 节报告里的数字填进项目"大语言模型微调（LoRA/QLoRA）"描述（示例仅为格式参考）：

> 自建 60 条英文指令评测集（概念解释/指令遵循/改写等 5 类），以 LLM-as-judge（0-5 盲评）对比 base 与微调后模型：指令遵循率 X% → Y%（+Z 个百分点），平均分 X → Y（提升约 Z%），优于 base 的题占比 W%（注：此处在 base 上用 facebook/opt-125m 完成端到端流程验证，报告与评测集留档可复现）。

> 若只想写一句：**微调后在自建 60 条指令评测集上指令遵循率 Y%，较微调前（base）提升约 Z%（LLM-as-judge 盲评，测试集与评分脚本留档可复现）**。

数字放简历的位置对应补测指南"六、填回简历的位置"表——与已有显存数据并列即可。

---

## 9. 附：命令速查与常见问题

```sh
# 全部命令在 微调评测方法A/ 文件夹内执行 (脚本按自身位置自动定位路径)

# ① 生成评测集(当前 60 条; 改题后重跑即覆盖 evalset/test.jsonl)
python create_testset.py

# ② base 与 ft 同题生成 (产物 outputs/base.jsonl, outputs/ft.jsonl)
python eval_generate.py

# ③ LLM 盲评打分 (密钥放同目录 .env, 认 LLM_API_KEY 或 DEEPSEEK_API_KEY)
# 默认走 .env 网关(qwen3-max-preview); 想用 DeepSeek: python eval_judge.py --model deepseek-chat --base-url https://api.deepseek.com/v1
python eval_judge.py

# ④ 出报告 (产物 outputs/eval_report.md)
python eval_report.py
```

| 常见问题 | 处理 |
|---|---|
| base 模型下载失败 | 脚本已设 `HF_ENDPOINT=https://hf-mirror.com`；仍失败检查网络代理 |
| 生成全是空串/重复字 | 题对 125m 太难或温度过低；把 `temperature` 提到 0.7、加 `repetition_penalty`，或换更简单的题 |
| judge 频繁失败 / 限流 | 调大 `time.sleep` 间隔；换 `--base-url`（如兼容网关）；失败 3 次的记录标 -1 会在报告中剔除 |
| 提示"未找到 API Key" | 在脚本同目录 `.env` 填 `DEEPSEEK_API_KEY` 或 `LLM_API_KEY`（不带引号）；或设同名环境变量 |
| 想换裁判模型/换 DeepSeek | 直接改 `.env` 的 `LLM_MODEL_NAME`/`LLM_BASE_URL`，或运行时 `--model` / `--base-url` 覆盖 |
| 想换成中文场景 | 换训练数据与评测题语言后全流程不变；但 opt-125m 中文能力弱，不建议拿它出中文数字 |
| 想把评测集扩到 100 条 | 每类按 3.3 风格续写，注意去重与难度，勿引入训练集原句 |
| 想让数字更有说服力 | 用同一套评测集/脚本跑 7B（如仓库 `Qwen2-7B` 链路），分数绝对值与提升都会更可观 |

---

*本文不再内嵌脚本代码：四个脚本均已落盘为独立文件（与本文同目录）`create_testset.py` / `eval_generate.py` / `eval_judge.py` / `eval_report.py`，正文各处"引用"的脚本以同名 .py 文件为准（可编辑、可独立运行）。产物目录 `evalset/`、`outputs/` 与 `test.jsonl`/`*.jsonl`/`eval_report.md` 就是面试时可当场打开的复现证据链。*
