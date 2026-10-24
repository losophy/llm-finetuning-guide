# Unsloth 微调实战指南

> 官方 Qwen2.5 (7B) 微调教程是一个 notebook：点开、从上到下跑完，就是一次完整微调。

---

## 一、唯一入口：官方 Notebook

点开下面这个链接（自动打开 Google Colab），运行时换成 **T4 GPU**，然后**从上到下逐个格子跑，一行代码都不要改**：

```
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb
```

这个 notebook 里只有四件事：

1. **装 Unsloth**（第一格自动装，不用管）
2. **下数据**（`unsloth/alpaca-cleaned`，英文 Alpaca 指令集，自动下载）
3. **训练**（默认 `max_steps=60`，几分钟跑完）
4. **推理测试**（训完当场问几句）

跑完 = 完成一次官方微调。

> 官方 notebook 是 Colab 版。要在 **Kaggle** 上跑：`File → Import Notebook → GitHub`，把链接里的 `colab.research.google.com/github/` 换成 `github.com/`，再按下面「三、Kaggle 特有步骤」补一句单卡限制即可，其余不动。

---

## 二、换数据集：只改一行（可选）

官方 notebook 里找到这一行：

```python
dataset = load_dataset("unsloth/alpaca-cleaned", split = "train")
```

把数据集名换掉即可，例如换成中文 Alpaca：

```python
dataset = load_dataset("shibing624/alpaca-zh", split = "train")
```

**其余一行都不动。**

前提：新数据集的字段得是 Alpaca 格式的 `instruction` / `input` / `output` 三列。字段名不同，就同步改 notebook 里的 `formatting_prompts_func` 那几行（它写死了 `examples["instruction"]` 等）。

---

## 三、Kaggle 特有步骤（走 Colab 可跳过）

官方 notebook 默认单卡（Colab 只有一张 T4）。Kaggle 会给你 **两张 T4**，不限制的话 Unsloth 会自动把模型切开跨两张卡跑（张量并行）——既慢又白占额度。所以导入 notebook 后，做两件事：

1. 右侧设置：Accelerator 选 `GPU T4 ×2`，**打开 Internet**（账号需已验证）
2. **在整份 notebook 的第一个代码格**加一句：

```python
%env CUDA_VISIBLE_DEVICES=0
```

> ⚠️ 这一句必须排在任何 `import torch` 之前。`CUDA_VISIBLE_DEVICES` 只在 CUDA **首次初始化**时生效，跑晚了就无效，只能重启会话。判断：加载模型时 Unsloth 横幅应显示 `Num GPUs = 1`；若显示 `Num GPUs = 2` 或 `output head -> cuda:1`，就是限晚了。

---

## 四、导出 GGUF + 本地 Ollama（官方 notebook 不含，实测补全）

官方 notebook 到"推理测试"为止。要部署到本地 Ollama，补下面三步（都是实测踩过坑的）。

### 4.1 导出 GGUF（云端）

```python
model.save_pretrained_gguf(
    "./qwen2.5-gguf",
    tokenizer,
    quantization_method="q4_k_m"  # Unsloth 默认
)
```

产物文件名**以模型名为前缀**：`Qwen2.5-7B.Q4_K_M.gguf`（不是 `unsloth.Q4_K_M.gguf`），约 **4.68 GB**（Windows 资源管理器按 2³⁰ 显示为 "4.36 GB"，是同一个文件）。

> ⚠️ **真正的瓶颈是磁盘不是内存**：导出时要同时放下「16-bit 权重 ≈15GB + f16 GGUF ≈15GB + q4 ≈4.7GB」约 30GB 中间产物，而 Kaggle 的 `/kaggle/working` 只有约 18.7GB → Unsloth 会**自动改写到 `/tmp`** 并打印 `working directory only has 18.7GB free ... goes to /tmp/unsloth_saves/...`。**`/tmp` 会话一停就回收、不进 Output**，所以导出后必须把它搬回 `/kaggle/working/`：

```bash
!mkdir -p /kaggle/working/gguf
!cp /tmp/unsloth_saves/qwen2.5-gguf_gguf/Qwen2.5-7B.Q4_K_M.gguf /kaggle/working/gguf/
```

然后右上角 **Save Version → Quick Save**（别选 "Save & Run All"，会整本重跑）。

### 4.2 下载到本地（HuggingFace 中转）

Kaggle 直链依赖会话登录态，外部下载器（FDM/IDM）常报 403。正解是**用 HuggingFace 中转**：

**① 云端上传**（传已生成的文件，不重新导出）：

```python
!pip install -q -U huggingface_hub
from huggingface_hub import login, HfApi
login()   # 设备授权，浏览器打开 https://hf.co/oauth/device 输验证码

api = HfApi()
api.create_repo("your-username/qwen2.5-7b-gguf", repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="/kaggle/working/gguf/Qwen2.5-7B.Q4_K_M.gguf",
    path_in_repo="Qwen2.5-7B.Q4_K_M.gguf",
    repo_id="your-username/qwen2.5-7b-gguf",
)
```

> ⚠️ token 必须是 **`Write`** 类型（`Read` 上传报 403）。别用 `push_to_hub_gguf()`——它会把整个导出重跑一遍。纯上传不需要 GPU，新建 notebook 时 Accelerator 选 `None` 即可。

**② 本地下载**：

```bash
pip install -U "huggingface_hub[hf_transfer]"
# PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
hf download your-username/qwen2.5-7b-gguf --local-dir D:/models
```

断了重跑同一条命令，会续传。

### 4.3 本地 Ollama 运行

建 `Modelfile`：

```
FROM D:/models/Qwen2.5-7B.Q4_K_M.gguf

TEMPLATE """### Instruction:
{{ .Prompt }}

### Response:
"""

PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
```

```bash
ollama create qwen2.5-finetuned -f Modelfile
ollama run qwen2.5-finetuned
```

> ⚠️ **`TEMPLATE` 和两条 `STOP` 都不能省**：训练用的是 Alpaca 格式（`### Instruction:`），而 GGUF 内嵌默认模板是 Qwen 的 ChatML。不指定 `TEMPLATE` 就格式错配、答非所问；不加 `STOP` 会自己往下编下一轮 `### Instruction:`。**不要加 `SYSTEM`**——训练数据没有 system 段。

---

## 五、踩过的坑（都实测过，报错时来查）

| 报错 / 现象 | 原因 | 解法 |
|---|---|---|
| `Num GPUs = 2` / `output head -> cuda:1` | Kaggle 没限单卡 | 首格补 `%env CUDA_VISIBLE_DEVICES=0`，重启会话 |
| `Some modules are dispatched on the CPU or the disk` | 会话没拿到 GPU，或显存被上一个模型占着 | 确认 Accelerator=GPU、查 GPU 额度，重启会话（`Run → Restart & Clear Output`） |
| 训练明显慢（远超 ≈5.7 s/step） | T4 上精度被误判成 bf16，实为 fp32 模拟 | 精度开关按**算力**写：`fp16=torch.cuda.get_device_capability()[0] < 8`（T4 算力 7.5 → fp16）；别用 `torch.cuda.is_bf16_supported()`（新版默认把软件模拟也算支持，T4 上误报 True） |
| `CUDA out of memory` | 显存不足 | `per_device_train_batch_size` 2→1，`gradient_accumulation_steps` 4→8；确认 `load_in_4bit=True` |
| `ollama create` 报 500，路径里出现 `manifests\D:models\...\latest` | `FROM` 文件名写错，Ollama 误当模型名解析 | 核对 `FROM` 指向的文件名与实际 `.gguf` 一致 |
| GGUF 落在 `/tmp` 不见了 | 磁盘不够，Unsloth 自动改落临时目录 | 见 4.1，`cp` 回 `/kaggle/working/` 再 Quick Save |

---

## 附：官方资源

- Notebook 仓库：<https://github.com/unslothai/notebooks>
- 官方微调指南：<https://unsloth.ai/docs/get-started/fine-tuning-llms-guide>
- 官方文档：<https://unsloth.ai/docs>
