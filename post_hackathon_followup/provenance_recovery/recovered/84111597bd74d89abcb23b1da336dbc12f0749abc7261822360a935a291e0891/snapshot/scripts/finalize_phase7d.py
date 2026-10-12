"""Build Phase 7D comparison artifacts from preserved raw measurements."""
from __future__ import annotations

import csv, json, math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT / "artifacts"
S_PARAMS, L_PARAMS = 8_923_392, 15_949_760


def rows(name):
    with (ART / name).open() as f: return list(csv.DictReader(f))


def closest(data, nominal): return min(data, key=lambda r: abs(int(r["training_tokens"])-nominal))


def slope(a, b):
    return (float(a["common_validation_loss"])-float(b["common_validation_loss"])) / ((int(b["training_tokens"])-int(a["training_tokens"]))/1e6)


def interpolate_tokens(data, seconds):
    for a, b in zip(data, data[1:]):
        x, y = float(a["cumulative_training_seconds"]), float(b["cumulative_training_seconds"])
        if x <= seconds <= y:
            f=(seconds-x)/(y-x); return int(round(int(a["training_tokens"])+f*(int(b["training_tokens"])-int(a["training_tokens"]))))
    return None


def write_csv(path, fields, data):
    with (ART/path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(data)


def main():
    s, l = rows("co4_s_25m_curve.csv"), rows("co4_l_data_rich_curve.csv")
    s10,s25,l10,l25=(closest(s,10_000_000),closest(s,25_000_000),closest(l,10_000_000),closest(l,25_000_000))
    fields=["model","parameters","training_tokens","tokens_per_parameter","common_validation_loss","common_validation_perplexity","tokens_per_second","cumulative_training_seconds","peak_rss_bytes","checkpoint_sha256"]
    def comp(model,p,r): return {"model":model,"parameters":p,"training_tokens":r["training_tokens"],"tokens_per_parameter":int(r["training_tokens"])/p,"common_validation_loss":r["common_validation_loss"],"common_validation_perplexity":r["common_validation_perplexity"],"tokens_per_second":r["tokens_per_second"],"cumulative_training_seconds":r["cumulative_training_seconds"],"peak_rss_bytes":r["peak_rss_bytes"],"checkpoint_sha256":r["checkpoint_sha256"]}
    write_csv("co4_s_vs_l_10m.csv",fields,[comp("Co4-S",S_PARAMS,s10),comp("Co4-L",L_PARAMS,l10)])
    write_csv("co4_s_vs_l_25m.csv",fields,[comp("Co4-S",S_PARAMS,s25),comp("Co4-L",L_PARAMS,l25)])
    curve=[]
    for model,p,data in (("Co4-S",S_PARAMS,s),("Co4-L",L_PARAMS,l)):
        for r in data: curve.append({"model":model,"parameters":p,"training_tokens":r["training_tokens"],"tokens_per_parameter":int(r["training_tokens"])/p,"common_validation_loss":r["common_validation_loss"],"common_validation_perplexity":r["common_validation_perplexity"],"cumulative_training_seconds":r["cumulative_training_seconds"],"tokens_per_second":r["tokens_per_second"]})
    write_csv("data_rich_scaling_curves.csv",list(curve[0]),curve)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for model,data in (("Co4-S",s),("Co4-L",l)):
        tokens=[int(r["training_tokens"])/1e6 for r in data]
        wall=[float(r["cumulative_training_seconds"])/60 for r in data]
        loss=[float(r["common_validation_loss"]) for r in data]
        ppl=[float(r["common_validation_perplexity"]) for r in data]
        axes[0,0].plot(tokens,loss,"o-",label=model); axes[0,1].plot(wall,loss,"o-",label=model)
        axes[1,0].plot(tokens,ppl,"o-",label=model); axes[1,1].plot(wall,ppl,"o-",label=model)
    labels=(("Validation loss","Training tokens (millions)"),("Validation loss","Training time (minutes)"),("Validation perplexity","Training tokens (millions)"),("Validation perplexity","Training time (minutes)"))
    for ax,(ylabel,xlabel) in zip(axes.flat,labels): ax.set(xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.25); ax.legend()
    fig.suptitle("Co4-S vs Co4-L data-rich scaling"); fig.tight_layout(); fig.savefig(ART/"data_rich_scaling_curves.png",dpi=160); plt.close(fig)
    wiki={m:json.loads((ART/p).read_text()) for m,p in (("Co4-S","wikitext_s_25m.json"),("Co4-L","wikitext_l_25m.json"))}
    write_csv("wikitext_bpb_results.csv",["model","tokens","perplexity","bits_per_byte","utf8_bytes","tokens_scored","context_length","stride","bos_policy","byte_counting_policy","checkpoint_sha256"],[{"model":m,"tokens":25000000,"perplexity":v["perplexity"],"bits_per_byte":v["bits_per_byte"],"utf8_bytes":v["utf8_bytes"],"tokens_scored":v["tokens_scored"],"context_length":v["context_length"],"stride":v["stride"],"bos_policy":v["bos_policy"],"byte_counting_policy":v["byte_counting_policy"],"checkpoint_sha256":v["checkpoint_sha256"]} for m,v in wiki.items()])
    gibc=[]
    for model,path in (("Co4-S","gibc_s_25m_raw.json"),("Co4-L","gibc_l_25m_raw.json")):
        d=json.loads((ART/path).read_text())
        for task in ("hellaswag","arc_easy","piqa","winogrande"):
            x=d["results"][task]; gibc.append({"model":model,"tokens":25000000,"task":task,"samples":x["sample_len"],"accuracy":x.get("acc,none"),"accuracy_stderr":x.get("acc_stderr,none"),"normalized_accuracy":x.get("acc_norm,none"),"normalized_accuracy_stderr":x.get("acc_norm_stderr,none"),"fewshot":0,"lm_eval_version":"0.4.13","wall_seconds":d["phase6_metadata"]["wall_seconds"],"checkpoint_sha256":d["phase6_metadata"]["checkpoint_sha256"]})
    write_csv("gibc_25m_comparison.csv",list(gibc[0]),gibc)
    gap10=float(s10["common_validation_loss"])-float(l10["common_validation_loss"]); gap25=float(s25["common_validation_loss"])-float(l25["common_validation_loss"])
    slopes=[]
    for model,data in (("Co4-S",s),("Co4-L",l)):
        for a,b in zip(data,data[1:]):
            if int(a["nominal_tokens"])>=7_500_000: slopes.append((model,int(a["nominal_tokens"]),int(b["nominal_tokens"]),slope(a,b)))
    points=(1,3,5,7.5,10,15,20,25)
    s_values=", ".join(f'{float(closest(s,int(x*1e6))["common_validation_loss"]):.4f}' for x in points)
    l_values=", ".join(f'{float(closest(l,int(x*1e6))["common_validation_loss"]):.4f}' for x in points)
    slope_rows="".join(f"| {m} | {a/1e6:g}M→{b/1e6:g}M | {v:.6f} |\n" for m,a,b,v in slopes)
    analysis=f"""# Data-rich scaling analysis

Observed points only; 25M is still an early scaling point and no asymptote is claimed.

```mermaid
xychart-beta
    title "Common validation loss vs training tokens"
    x-axis "million tokens" [1, 3, 5, 7.5, 10, 15, 20, 25]
    y-axis "loss" 3.3 --> 4.6
    line [{s_values}]
    line [{l_values}]
```

The first line is Co4-S and the second is Co4-L. The source CSV supports the required loss and perplexity plots against tokens and cumulative wall time.

![Loss and perplexity against tokens and wall time](data_rich_scaling_curves.png)

| Model | Interval | loss improvement / M tokens |
|---|---:|---:|
{slope_rows}
At 10M the S−L loss gap is **{gap10:.6f}**; at 25M it is **{gap25:.6f}**. The larger-model advantage grew by {gap25-gap10:.6f} loss ({(gap25/gap10-1)*100:.1f}% relative to the 10M gap), with a non-monotonic path at 15M. This supports result **A**, cautiously: the overall advantage grew materially, while the final 20M→25M slope was slightly better for S.

Equal-wall interpolation at the 10M comparison: in Co4-S's {float(s10['cumulative_training_seconds']):.1f}s, Co4-L reaches approximately {interpolate_tokens(l,float(s10['cumulative_training_seconds'])):,} tokens. In Co4-L's {float(l10['cumulative_training_seconds']):.1f}s, Co4-S reaches approximately {interpolate_tokens(s,float(l10['cumulative_training_seconds'])):,} tokens. These interpolations are descriptive, not observed checkpoints.
"""
    (ART/"data_rich_scaling_analysis.md").write_text(analysis)
    decision=f"""# Phase 7D decision\n\nDATA-RICH CAPACITY RESULT: **A — Co4-L's advantage grows materially with data overall.**\n\nThe common-validation gap favoring L grows from {gap10:.6f} at 10M to {gap25:.6f} at 25M. Co4-L also leads on WikiText PPL and BPB. The gap is not monotonic across every interval, so this is evidence for continued measurement rather than saturation.\n\nMODEL(S) ADVANCING TO 50M: **Co4-L; Co4-S optionally as the controlled reference because compute is inexpensive. No 50M run has started.**\n\nSHOULD ~24M MODEL BE TESTED YET: **NO.** Reconsider only after the 50M S-vs-L comparison.\n\nBROADER DATA EXPANSION PRIORITY: **HIGH.** Both models remain data-limited, and DATA-C's FineWeb slice is much smaller than the intended future scale.\n"""
    (ART/"phase7d_decision.md").write_text(decision)
    for model,p,r,w in (("s",S_PARAMS,s25,wiki["Co4-S"]),("l",L_PARAMS,l25,wiki["Co4-L"])):
        metrics={"tokens_trained":25000000,"parameters":p,"val_loss":float(r["common_validation_loss"]),"val_ppl":float(r["common_validation_perplexity"]),"wall_seconds":float(r["cumulative_training_seconds"]),"tokens_per_second":float(r["tokens_per_second"]),"peak_rss_bytes":int(r["peak_rss_bytes"]),"checkpoint_sha256":r["checkpoint_sha256"],"wikitext_103_perplexity":w["perplexity"],"wikitext_bits_per_byte":w["bits_per_byte"]}
        (ART/f"phase7d_{model}_25m_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    report=f"""# Phase 7D — Data-rich capacity scaling\n\nBoth controlled DATA-C lineages reached exactly 25M tokens with no Spot interruption. Co4-S resumed the canonical 10M state exactly; Co4-L used random initialization with the same predeclared seed and data ordering. No HPO, benchmark tuning, Triton integration, broader-data substitution, or 50M continuation occurred.\n\n| Metric | Co4-S | Co4-L |\n|---|---:|---:|\n| Parameters | {S_PARAMS:,} | {L_PARAMS:,} |\n| Tokens | 25,000,000 | 25,000,000 |\n| Tokens/parameter | {25e6/S_PARAMS:.4f} | {25e6/L_PARAMS:.4f} |\n| Common validation loss | {float(s25['common_validation_loss']):.6f} | {float(l25['common_validation_loss']):.6f} |\n| Common validation PPL | {float(s25['common_validation_perplexity']):.5f} | {float(l25['common_validation_perplexity']):.5f} |\n| Training tok/s | {float(s25['tokens_per_second']):.2f} | {float(l25['tokens_per_second']):.2f} |\n| Cumulative training time | {float(s25['cumulative_training_seconds']):.2f}s | {float(l25['cumulative_training_seconds']):.2f}s |\n| Peak RSS | {int(s25['peak_rss_bytes'])/2**30:.3f} GiB | {int(l25['peak_rss_bytes'])/2**30:.3f} GiB |\n| WikiText-103 PPL | {wiki['Co4-S']['perplexity']:.5f} | {wiki['Co4-L']['perplexity']:.5f} |\n| WikiText BPB | {wiki['Co4-S']['bits_per_byte']:.5f} | {wiki['Co4-L']['bits_per_byte']:.5f} |\n\nThe 10M S−L gap was {gap10:.6f}; the 25M gap is {gap25:.6f}. See `data_rich_scaling_analysis.md` for slopes and equal-wall interpolation. The final classification is **A**, with the qualification that the gap did not widen in every interval.\n\nRequired final fields\n\nCo4-S 25M COMMON VAL LOSS: {float(s25['common_validation_loss']):.6f}\n\nCo4-L 25M COMMON VAL LOSS: {float(l25['common_validation_loss']):.6f}\n\nCo4-S 25M TOK/S: {float(s25['tokens_per_second']):.2f}\n\nCo4-L 25M TOK/S: {float(l25['tokens_per_second']):.2f}\n\nCo4-S 25M TOKENS/PARAMETER: {25e6/S_PARAMS:.6f}\n\nCo4-L 25M TOKENS/PARAMETER: {25e6/L_PARAMS:.6f}\n\nDATA-RICH CAPACITY RESULT: A\n\nMODEL(S) ADVANCING TO 50M: Co4-L; optionally Co4-S as controlled reference\n\nSHOULD ~24M MODEL BE TESTED YET: NO\n\nBROADER DATA EXPANSION PRIORITY: HIGH\n"""
    (ART/"phase7d_report.md").write_text(report)


if __name__=="__main__": main()
