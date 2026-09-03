# -*- coding: utf-8 -*-
"""
LLM-as-judge 盲评打分
输入: 脚本所在目录 outputs/base.jsonl + outputs/ft.jsonl (同 id 对齐)
输出: 脚本所在目录 outputs/scores.json  (每条含 base_score / ft_score / follows_*)

用法:
  1) 设置环境变量 DEEPSEEK_API_KEY=<你的 key>
  2) python eval_judge.py [--model deepseek-chat] [--base-url https://api.deepseek.com/v1]
兼容任意 OpenAI 格式接口(改 base-url/model 即可, 如本地 vLLM)。
成本: 约 2*题数 次调用, 60 题约 120 次, 每次千余 token, 通常不足 1 元。
"""
import os, sys, json, time, random, argparse
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    user = (f"Instruction: {inst}\n"
            f"Reference answer (for content comparison, answers may differ but still be correct): {ref}\n"
            f"Model answer: {answer}")
    body = {"model": cfg.model, "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user}],
            "temperature": 0, "max_tokens": 200}
    for attempt in range(3):
        try:
            r = requests.post(f"{cfg.base_url}/chat/completions",
                              json=body,
                              headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                              timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            # 抽取 JSON 对象(容错: 模型可能夹带解释文字)
            s, e = content.find("{"), content.rfind("}")
            obj = json.loads(content[s:e+1])
            return {"score": int(obj["score"]),
                    "follows_instruction": bool(obj["follows_instruction"])}
        except Exception as ex:
            print(f"  第{attempt+1}次失败: {ex} | 原文片段: {content[:120] if 'content' in dir() else ''}", file=sys.stderr)
            time.sleep(2)
    return {"score": -1, "follows_instruction": False}   # 三次失败标记 -1, 报告时剔除

def main():
    out_dir = os.path.join(SCRIPT_DIR, "outputs")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(out_dir, "base.jsonl"))
    ap.add_argument("--ft",   default=os.path.join(out_dir, "ft.jsonl"))
    ap.add_argument("--out",  default=os.path.join(out_dir, "scores.json"))
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    args = ap.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY")

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
        res = judge_once(d["instruction"], d["reference"], d["answer"], args)
        scores.setdefault(i, {})[f"{who}_score"] = res["score"]
        scores.setdefault(i, {})[f"{who}_follows"] = res["follows_instruction"]
        print(f"[{n}/{len(tasks)}] id={i} {who}: {res['score']} follow={res['follows_instruction']}")
        time.sleep(0.3)                        # 温和限速, 避免触发限流

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": args.model, "rubric": "0-5", "judge": "blind"},
                   "items": [{"id": i, **scores[i]} for i in ids]},
                  f, ensure_ascii=False, indent=2)
    print(f"已写入 {args.out}")

if __name__ == "__main__":
    main()
