# -*- coding: utf-8 -*-
"""
双模型同题批量推理: base (facebook/opt-125m) vs 微调合并模型 (上级目录 opt-125m-merged)
输出: 脚本所在目录 outputs/base.jsonl, outputs/ft.jsonl  (与 test.jsonl 的 id 一一对应)

用法: python eval_generate.py
环境: 与你训练/推理一致 (torch + transformers 5.x)
首次运行会自动下载 base 模型 (~250MB), 已设 hf-mirror
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "facebook/opt-125m"
FT_MODEL   = os.path.join(os.path.dirname(SCRIPT_DIR), "opt-125m-merged")  # 上级目录的合并模型; 换位置就改这里
PROMPT_TEMPLATE = "### Human: {instruction}\n### Assistant: "

# 统一解码参数: 两边完全一致, 固定种子保证可复现
GEN = dict(do_sample=True, temperature=0.5, top_p=0.9,
           top_k=50, repetition_penalty=1.05, max_new_tokens=128)
SEED = 42

def load(path_or_name, tag):
    t0 = time.time()
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path_or_name, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(path_or_name)
    model.to("cuda" if use_cuda else "cpu").eval()
    print(f"[{tag}] 加载完成 {time.time()-t0:.1f}s | 设备 {'cuda' if use_cuda else 'cpu'} | {path_or_name}")
    return model, tok

def generate_all(model, tok, rows, tag):
    results, t0 = [], time.time()
    torch.manual_seed(SEED)                # 同一批题共用同一种子(整批前设一次)
    for r in rows:
        prompt = PROMPT_TEMPLATE.format(instruction=r["instruction"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **GEN,
                                 pad_token_id=tok.eos_token_id)
        new = out[0][inputs["input_ids"].shape[1]:]
        answer = tok.decode(new, skip_special_tokens=True).strip()
        results.append({"id": r["id"], "instruction": r["instruction"],
                        "reference": r["reference"], "answer": answer})
    print(f"[{tag}] {len(results)} 条生成完毕 {time.time()-t0:.1f}s")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=os.path.join(SCRIPT_DIR, "evalset", "test.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(SCRIPT_DIR, "outputs"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.testset, encoding="utf-8")]
    os.makedirs(args.outdir, exist_ok=True)
    print(f"评测集: {len(rows)} 条")

    # --- base ---
    m, t = load(BASE_MODEL, "base")
    base = generate_all(m, t, rows, "base")
    del m, t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 微调后(合并模型) ---
    m, t = load(FT_MODEL, "ft")
    ft = generate_all(m, t, rows, "ft")

    for tag, data in [("base", base), ("ft", ft)]:
        p = os.path.join(args.outdir, f"{tag}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"已写入 {p}")

    # 抽样打印 3 条便于肉眼检查
    print("\n--- 抽样对比 (前 3 条) ---")
    for b, f in zip(base[:3], ft[:3]):
        print(f"\n[Q] {b['instruction']}")
        print(f"[base] {b['answer'][:150]}")
        print(f"[ft]   {f['answer'][:150]}")

if __name__ == "__main__":
    main()
