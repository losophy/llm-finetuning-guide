# -*- coding: utf-8 -*-
"""
读取脚本所在目录 outputs/scores.json + evalset/test.jsonl, 计算指标, 输出 outputs/eval_report.md
用法: python eval_report.py
"""
import json, argparse, os, datetime
from collections import defaultdict

def avg(xs): return sum(xs) / len(xs) if xs else float("nan")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "outputs")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(out_dir, "scores.json"))
    ap.add_argument("--testset", default=os.path.join(script_dir, "evalset", "test.jsonl"))
    ap.add_argument("--out", default=os.path.join(out_dir, "eval_report.md"))
    args = ap.parse_args()

    data   = json.load(open(args.scores, encoding="utf-8"))
    items  = data["items"]
    meta   = data.get("meta", {})
    cats   = {d["id"]: d["category"] for d in map(json.loads, open(args.testset, encoding="utf-8"))}

    valid = [it for it in items if it["base_score"] >= 0 and it["ft_score"] >= 0]
    print(f"有效条数: {len(valid)} / {len(items)}")

    def metric(valid):
        bs = [i["base_score"] for i in valid]; fs = [i["ft_score"] for i in valid]
        bf = [i["base_follows"] for i in valid]; ff = [i["ft_follows"] for i in valid]
        win = sum(1 for i in valid if i["ft_score"] > i["base_score"])
        return dict(
            base_avg=avg(bs), ft_avg=avg(fs),
            base_follow=avg(bf), ft_follow=avg(ff), win_rate=win/len(valid),
            n=len(valid))

    overall = metric(valid)
    by_cat = {}
    for c in sorted(set(cats[i["id"]] for i in valid)):
        sub = [i for i in valid if cats[i["id"]] == c]
        by_cat[c] = metric(sub)

    today = datetime.date.today().isoformat()
    judge = meta.get("model", "__填写__")
    lines = ["# 微调评测报告 (方法A: 自建评测集 + LLM 打分)", "",
             f"- 日期: {today} | 设备: __填写__ | 评测模型: base=`facebook/opt-125m` vs ft=`opt-125m-merged`",
             f"- 有效条数: {overall['n']} | judge: {judge} | 打分: 0-5 盲评 | 空答案按 0 分计", "",
             "## 总指标", "",
             "| 指标 | base | ft | 提升 |",
             "|---|---|---|---|",
             f"| 平均分 | {overall['base_avg']:.2f} | {overall['ft_avg']:.2f} | +{overall['ft_avg']-overall['base_avg']:.2f} ({(overall['ft_avg']-overall['base_avg'])/overall['base_avg']*100:+.1f}%) |",
             f"| 指令遵循率 | {overall['base_follow']*100:.1f}% | {overall['ft_follow']*100:.1f}% | +{(overall['ft_follow']-overall['base_follow'])*100:.1f} 个百分点 |",
             f"| 胜出率 (ft>base) | - | - | {overall['win_rate']*100:.1f}% |", "",
             "## 分项 (按类别)", "",
             "| 类别 | n | base 均分 | ft 均分 | 提升 | base 遵循 | ft 遵循 |",
             "|---|---|---|---|---|---|---|"]
    for c, m in by_cat.items():
        lines.append(f"| {c} | {m['n']} | {m['base_avg']:.2f} | {m['ft_avg']:.2f} "
                     f"| {m['ft_avg']-m['base_avg']:+.2f} | {m['base_follow']*100:.0f}% | {m['ft_follow']*100:.0f}% |")
    lines += ["", "> 说明: 以上数字均来自实测, 保留本目录 outputs/ 下 jsonl 作为复现证据。"]

    open(args.out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n报告已写入 {args.out}")

if __name__ == "__main__":
    main()
