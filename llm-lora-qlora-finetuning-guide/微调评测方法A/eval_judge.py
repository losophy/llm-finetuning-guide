# -*- coding: utf-8 -*-
"""
LLM-as-judge 盲评打分
输入: 脚本所在目录 outputs/base.jsonl + outputs/ft.jsonl (同 id 对齐)
输出: 脚本所在目录 outputs/scores.json  (每条含 base_score / ft_score / follows_*)

用法:
  1) 在脚本同目录的 .env 中填写密钥, 支持两套命名(任选其一):
     - LLM_API_KEY / LLM_MODEL_NAME / LLM_BASE_URL   (OpenAI 兼容网关通用, 如百炼 qwen)
     - DEEPSEEK_API_KEY                                (DeepSeek 官方, model/base-url 走默认)
  2) python eval_judge.py [--model xxx] [--base-url https://.../v1]   # 显式参数优先于 .env
兼容任意 OpenAI 格式接口。成本: 约 2*题数 次调用, 60 题约 120 次, 每次千余 token, 通常不足 1 元。
"""
import os, sys, json, time, random, argparse
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_env_file(path):
    """极简 .env 解析: KEY=VALUE, 支持 # 注释与引号, 不覆盖已存在的环境变量."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k:
                os.environ.setdefault(k, v)

load_env_file(os.path.join(SCRIPT_DIR, ".env"))

def resolve_cfg(args):
    """密钥/模型/base-url 解析顺序: 显式命令行参数 > .env(LLM_*) > DeepSeek 默认."""
    args.api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
    args.model = args.model or os.environ.get("LLM_MODEL_NAME") or "deepseek-chat"
    args.base_url = args.base_url or os.environ.get("LLM_BASE_URL") or "https://api.deepseek.com/v1"
    return args

JUDGE_SYSTEM = """You are a strict and fair evaluator for instruction-following quality.
Score the model answer 0-5 using this rubric:
5 = fully correct and satisfies every hard constraint of the instruction (or equivalent to the reference)
4 = correct, mostly satisfies constraints, minor flaw
3 = on the right track, partly correct or only partly follows constraints
2 = some relevant content but clearly wrong or ignores main constraints
1 = severely off-topic or nearly empty
0 = empty, garbled, or unrelated
Also set "follows_instruction" to true only if ALL hard constraints of the instruction are met (e.g. exact number of items, one sentence, Yes/No only, word limit, required starting words). If the instruction has no hard constraint, set it to true. Content correctness does NOT affect follows_instruction.
Output JSON only: {"score": <int 0-5>, "follows_instruction": <true/false>, "comment": "<short reason in English>"}"""

def judge_once(inst, ref, answer, cfg):
    # 答案用标记包裹, 避免空串/首尾空白造成裁判误读(实测空答案会被误评 5 分)
    user = (f"Instruction: {inst}\n"
            f"Reference answer (for content comparison, answers may differ but still be correct): {ref}\n"
            f"Model answer: <answer>{answer}</answer>")
    body = {"model": cfg.model, "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user}],
            "temperature": 0, "max_tokens": 200}
    for attempt in range(3):
        try:
            r = requests.post(f"{cfg.base_url}/chat/completions",
                              json=body,
                              headers={"Authorization": f"Bearer {cfg.api_key}"},
                              timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            # 抽取 JSON 对象(容错: 模型可能夹带解释文字)
            s, e = content.find("{"), content.rfind("}")
            obj = json.loads(content[s:e+1])
            return {"score": int(obj["score"]),
                    "follows_instruction": bool(obj["follows_instruction"]),
                    "comment": str(obj.get("comment", ""))[:200]}
        except Exception as ex:
            print(f"  第{attempt+1}次失败: {ex} | 原文片段: {content[:120] if 'content' in dir() else ''}", file=sys.stderr)
            time.sleep(2)
    return {"score": -1, "follows_instruction": False, "comment": "judge call failed"}   # 三次失败标记 -1, 报告时剔除

def judge_answer(inst, ref, answer, cfg):
    """单条打分: 空答案不调 API, 按 rubric 直接记 0 分(empty, garbled, or unrelated)."""
    if not (answer or "").strip():
        return {"score": 0, "follows_instruction": False, "comment": "empty answer -> 0 by rubric"}
    return judge_once(inst, ref, answer, cfg)

def main():
    out_dir = os.path.join(SCRIPT_DIR, "outputs")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(out_dir, "base.jsonl"))
    ap.add_argument("--ft",   default=os.path.join(out_dir, "ft.jsonl"))
    ap.add_argument("--out",  default=os.path.join(out_dir, "scores.json"))
    ap.add_argument("--model", default=None, help="裁判模型; 默认取 .env 的 LLM_MODEL_NAME, 否则 deepseek-chat")
    ap.add_argument("--base-url", default=None, help="API base; 默认取 .env 的 LLM_BASE_URL, 否则 https://api.deepseek.com/v1")
    args = ap.parse_args()
    args = resolve_cfg(args)
    if not args.api_key:
        sys.exit("未找到 API Key: 请在脚本同目录 .env 中填写 DEEPSEEK_API_KEY 或 LLM_API_KEY, 或设置同名环境变量后重试")

    base = {d["id"]: d for d in map(json.loads, open(args.base, encoding="utf-8"))}
    ft   = {d["id"]: d for d in map(json.loads, open(args.ft,   encoding="utf-8"))}
    ids  = sorted(set(base) & set(ft))
    print(f"待打分: {len(ids)} 条 (base+ft 共 {len(ids)*2} 次调用)")

    # 盲评: 构建打分任务并打乱顺序
    tasks = []
    for i in ids:
        tasks.append(("base", i, base[i])); tasks.append(("ft", i, ft[i]))
    random.Random(42).shuffle(tasks)          # 固定洗牌种子, 保证可复现

    scores = {}
    for n, (who, i, d) in enumerate(tasks, 1):
        res = judge_answer(d["instruction"], d["reference"], d.get("answer", ""), args)
        tag = "EMPTY->0" if res["comment"].startswith("empty") else res["score"]
        scores.setdefault(i, {})[f"{who}_score"] = res["score"]
        scores.setdefault(i, {})[f"{who}_follows"] = res["follows_instruction"]
        scores.setdefault(i, {})[f"{who}_comment"] = res["comment"]
        print(f"[{n}/{len(tasks)}] id={i} {who}: {tag} follow={res['follows_instruction']}")
        time.sleep(0.3)                        # 温和限速, 避免触发限流

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": args.model, "base_url": args.base_url,
                            "rubric": "0-5", "judge": "blind",
                            "empty_as_zero": True},
                   "items": [{"id": i, **scores[i]} for i in ids]},
                  f, ensure_ascii=False, indent=2)
    print(f"已写入 {args.out}")

if __name__ == "__main__":
    main()
