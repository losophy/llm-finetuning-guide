# 同条件性能基准：Unsloth vs 原生 HuggingFace（Qwen2.5-7B）

跑出《Unsloth与原生HuggingFace对照.md》4.4 结果表要填的数字。

| 文件 | 用途 |
|---|---|
| `benchmark_qwen7b.py` | 基准脚本，顶部 `SIDE` 一行切换两侧 |
| `README.md` | 跑法、口径、硬规则、结果记录表 |

**为什么只有一个脚本**：同条件对照最大的风险是"两侧配置悄悄漂移"。数据、超参、采集代码都放在同一个文件里物理共用，只有 `load_model()` 分支——口径不可能不一致。

---

## 一、怎么跑（Colab）

**两个独立 notebook，各跑一侧。** 顺序不限。

### 第 1 个 cell：环境准备（两个 notebook 必须用完全相同的这一格）

```python
!pip install unsloth
!pip install --upgrade --no-cache-dir torch trl peft accelerate bitsandbytes datasets
```

装完 **重启会话**（运行时 → 重启会话），两个 notebook 都这么做。

### 第 2 个 cell：把 `benchmark_qwen7b.py` 整段粘进来

只改顶部这一行：

```python
SIDE = "unsloth"   # 第一个 notebook 填 "unsloth"
```

第二个 notebook 填 `SIDE = "native"`，其余一字不改。

跑完把控制台输出的那段（设备 / 版本 / 可训练参数 / 四项指标）抄进第六节的表。

> Colab 免费档给的是 **T4 16GB**。Qwen2.5-7B 的 4-bit QLoRA 在 T4 上两侧都能跑，但原生侧更吃显存，遇到 OOM 看第五节。

---

## 二、口径（决定对照有没有意义）

两侧**完全一致**，且都写在脚本的公共常量区：

| 项 | 值 |
|---|---|
| 基座 | `Qwen/Qwen2.5-7B`（两侧同一份权重，各自加载时做 4-bit nf4 量化） |
| 数据 | `yahma/alpaca-cleaned`，`shuffle(seed=42).select(range(2000))` |
| 模板 | Alpaca `### Instruction / ### Input / ### Response` |
| 最大长度 | 2048 |
| 步数 | `max_steps=60`（**不用 epochs**） |
| batch / 累积 | 2 / 4 |
| 学习率 | 2e-4，cosine，warmup 10 步 |
| 优化器 | `adamw_8bit` |
| LoRA | r=16 / alpha=32 / dropout=0 / `target_modules` 7 个（q,k,v,o,gate,up,down） |
| 精度 | fp16（T4 无 bf16） |
| eval | 两侧都关闭 |

### 几条容易踩的

1. **不用 `unsloth/Qwen2.5-7B-bnb-4bit`。** 预量化权重和加载时量化可能不完全等价，会被质疑"权重来源不一致"。两侧都用官方 `Qwen/Qwen2.5-7B`。
2. **`target_modules` 必须都是 7 个。** 仓库里 `finetune_basic.py` 是 7 个、原生脚本是 4 个——直接拿那两个脚本比，可训练参数量不同，耗时不可比。
3. **精度写死 fp16。** 别用 `torch.cuda.is_bf16_supported()` 自动判断，换到 L4/A100 两侧就会不一致。
4. **原生侧不装 Flash Attention 2**，用 PyTorch 默认 SDPA。装 FA2 会改变对比口径。
5. **梯度检查点两侧对称**：Unsloth 用 `use_gradient_checkpointing="unsloth"`，原生用 `gradient_checkpointing_enable()`。这是被测量的差异本身，不算不公。
6. **显存口径**：`reset_peak_memory_stats()` 在 `train()` 之前调用，所以报的是**训练阶段峰值，不含模型加载**。填表时别当成整机峰值。

---

## 三、三条硬规则

1. **两侧必须独立 notebook / 独立进程。** `import unsloth` 会在导入时全局改写 HF 的模型实现，同一会话里连跑两侧，原生侧的数字作废。
2. **两侧必须同一型号 GPU。** 脚本会打印 `torch.cuda.get_device_name(0)`；T4 与 L4 的数字不可混在一张表里。Colab 分配是随机的，两次跑之前都确认一下。
3. **跑 2~3 次取平均。** 单次数字有噪声，尤其耗时。

另外两侧打印的库版本（torch / transformers / trl / peft / bitsandbytes）也应一致——这就是为什么要用同一格 install cell。

---

## 四、降档预案

只有一条铁律：**要降就两侧一起降，绝不能只降一边。**

| 症状 | 处理 |
|---|---|
| 原生侧 OOM，Unsloth 侧正常 | 两侧同时把 `MAX_SEQ_LEN` 降到 1024、`BATCH_SIZE` 降到 1、`GRAD_ACCUM` 调到 8 |
| 还 OOM | 两侧同时降到 `MAX_SEQ_LEN=512` |
| 7B 两侧都跑不动 | 两侧一起换 `Qwen/Qwen2.5-3B`，并在表下注明换了模型 |
| Colab 会话中断 | `MAX_STEPS` 调小（如 30）后两侧重跑；两侧步数必须一致 |

---

## 五、结果记录表（照抄进对照文档 4.4）

| 指标 | Unsloth | 原生 HF | 倍数 / 差异 |
|---|---|---|---|
| 训练耗时（s） | | | |
| 峰值显存（GB） | | | |
| 每秒样本数 | | | |
| 可训练参数量 | | | |

**环境脚注（每次都要填）**

- GPU 型号：______（`torch.cuda.get_device_name(0)`）
- 库版本：torch ___ / transformers ___ / trl ___ / peft ___ / bitsandbytes ___
- `max_steps=60`，`max_length=2048`，batch 2 × accum 4
- `target_modules`：q,k,v,o,gate,up,down（7 个）
- 跑了几次、是否取平均：______
- 「可训练参数量」两侧应完全相同——不一致说明口径漂了，数字作废

---

## 六、已知限制

- **`Qwen/Qwen2.5-7B` 是 fp16 权重（约 15GB）**，两个 notebook 各自要下载一次，比用 unsloth 预量化仓慢。这是为了"同一份权重"付的代价。若确实嫌慢，可以两侧**都**改成 `unsloth/Qwen2.5-7B-bnb-4bit`（原生侧也能加载），但绝不允许只改一侧。
- 单卡对比，不含多卡 / DeepSpeed 场景。
- Colab 免费档的 GPU 型号会变，跨天数据不要直接横向比较。
