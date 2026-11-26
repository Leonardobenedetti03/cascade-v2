"""Build Tables 1, 2 and 3 of the report from the files in results/.

Table 1 (RQ1): accuracy and compute of CamFormer, EgoVLP, the oracle gate, the class rule and the cascade.
Table 2 (RQ2): the gate ablation, from the threshold on p_max to LightGBM on all the features.
Table 3: CamFormer's accuracy and the cascade's escalation, class by class.

It only reads results/nested_cv/report_n38_cal/ (and, for the labels of Table 3,
results/features/official_v2/heads/camformer/), and writes to results/report/ one CSV per table, with full
precision, and tables.md, with the tables rounded as in the report. Nothing is rerun.

Run it from the repository root:
    python scripts/06_make_report_tables.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

RUN = Path("results/nested_cv/report_n38_cal")
PRED = Path("results/features/official_v2/heads/camformer")
OUT = Path("results/report")
U = "pretrain_unseen"  # takes CamFormer was not trained on: the main subset of the report
BETA = 0.05  # the summary point of the cost rule
CLASSES = ["Basketball", "Dance", "Soccer", "Cooking", "Health", "Bike repair"]  # order of the heads' outputs


def metrics_with_cost(run: Path) -> pd.DataFrame:
    m = pd.read_csv(run / "metrics.csv")
    c = pd.read_csv(run / "cost_vs_egovlp.csv")
    key = ["subset", "system", "rule", "param"]
    return m.merge(c[key + ["cost_frac_of_egovlp_only", "cost_frac_ci_low", "cost_frac_ci_high"]], on=key)


def pick(df: pd.DataFrame, **match) -> pd.Series:
    """The single row matching all the given columns. Anything else means the table is built wrong."""
    q = df
    for col, val in match.items():
        q = q[np.isclose(q[col], val)] if isinstance(val, float) else q[q[col] == val]
    if len(q) != 1:
        raise LookupError(f"expected 1 row for {match}, found {len(q)}")
    return q.iloc[0]


def table1(mc: pd.DataFrame) -> pd.DataFrame:
    systems = [
        ("CamFormer only", "always_L0", "fixed", 0.0),
        ("EgoVLP only", "always_L1", "fixed", 0.0),
        ("Oracle gate", "oracle", "fixed", 0.0),
        ("Class rule", "class_rule", "natural", 0.0),
        ("Cascade (threshold)", "thr_maxprob", "cost", BETA),
    ]
    rows = []
    for name, system, rule, param in systems:
        a = pick(mc, subset=U, system=system, rule=rule, param=param)
        b = pick(mc, subset="all", system=system, rule=rule, param=param)
        rows.append(
            {
                "system": name,
                "acc": a.acc,
                "acc_ci_low": a.acc_ci_low,
                "acc_ci_high": a.acc_ci_high,
                "bacc": a.bacc,
                "compute_pct": 100 * a.cost_frac_of_egovlp_only,
                "compute_ci_low": 100 * a.cost_frac_ci_low,
                "compute_ci_high": 100 * a.cost_frac_ci_high,
                "acc_all": b.acc,
                "compute_pct_all": 100 * b.cost_frac_of_egovlp_only,
            }
        )
    return pd.DataFrame(rows)


def table2(run: Path, mc: pd.DataFrame) -> pd.DataFrame:
    auc = pd.read_csv(run / "auc.csv")
    cmp_ = pd.read_csv(run / "comparisons.csv")
    curves = pd.read_csv(run / "curves.csv")
    n_unseen = int(pd.read_csv(run / "models.csv").set_index("subset").loc[U, "n_takes"])
    gates = [
        # gate, model, features, the gate it is compared with (Delta AUROC = this gate minus that one)
        ("thr_maxprob", "Threshold", "p_max", None),
        ("logreg_A", "Logistic", "A", "thr_maxprob"),
        ("logreg_AC", "Logistic", "A+C", "logreg_A"),
        ("logreg_ABC", "Logistic", "A+B+C", "logreg_AC"),
        ("lgbm_ABC", "LightGBM", "A+B+C", None),
        ("class_rule", "Class rule", "class", "thr_maxprob"),
    ]
    rows = []
    for g, model, feats, other in gates:
        a = pick(auc, subset=U, scorer=g)
        # the class rule has no price beta: it is shown at its own escalation share
        rule, param = ("natural", 0.0) if g == "class_rule" else ("cost", BETA)
        r = pick(mc, subset=U, system=g, rule=rule, param=param)
        cv = curves[(curves.subset == U) & (curves.scorer == g)]
        at30 = cv.loc[(cv.r - 30).abs().idxmin()]  # escalate the 30% of takes with the highest calibrated risk
        row = {
            "gate": model,
            "features": feats,
            "scorer": g,
            "auroc": a.auroc,
            "auroc_ci_low": a.auroc_ci_low,
            "auroc_ci_high": a.auroc_ci_high,
            "escalated_pct": 100 * r.rho,
            "acc": r.acc,
            "errors_left_at_30pct": (1 - at30.acc) * n_unseen,
            "delta_auroc": np.nan,
            "p_holm": np.nan,
        }
        if other is not None:
            # the comparisons file stores threshold minus class rule; the table shows class rule minus threshold
            a_, b_, sign = (g, other, 1) if g != "class_rule" else (other, g, -1)
            c = pick(cmp_, subset=U, a=a_, b=b_, metric="auroc")
            row["delta_auroc"], row["p_holm"] = sign * c["diff"], c.p_holm
        rows.append(row)
    return pd.DataFrame(rows)


def table3(run: Path, root: Path) -> pd.DataFrame:
    takes = pd.read_csv(run / "takes.csv.gz")
    pred = pd.concat(
        [pd.read_csv(root / PRED / "predictions.csv"), pd.read_csv(root / PRED / "sealed/predictions.csv")]
    )
    lab = pred.drop_duplicates("take_name").set_index("take_name")[["label", "camformer_pretrain"]]
    t = takes.join(lab, on="take_name")
    t = t[t.camformer_pretrain != "train"].copy()
    t["y"] = t.label.map({c: i for i, c in enumerate(CLASSES)})
    t["escalated"] = t.rup__thr_maxprob >= BETA  # the cost rule, on the conservative risk
    t["camformer_right"] = t.pred0 == t.y
    t["cascade_right"] = np.where(t.escalated, t.pred1 == t.y, t.camformer_right)
    n_repeats = t.repeat.nunique()
    rows = []
    for c in ["Dance", "Basketball", "Soccer", "Cooking", "Health", "Bike repair"]:  # physical, then procedural
        g = t[t.label == c]
        rows.append(
            {
                "class": c,
                "takes": g.take_name.nunique(),
                "camformer_acc": g.camformer_right.mean(),
                "escalated_pct": 100 * g.escalated.mean(),
                "errors_left_per_repeat": (~g.cascade_right).sum() / n_repeats,
            }
        )
    return pd.DataFrame(rows)


def dot(x: float, nd: int = 3) -> str:
    """Numbers below 1 without the leading zero, as in the report (.882)."""
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("0.") else s.replace("-0.", "−.").replace("-", "−")


def signed(x: float) -> str:
    return ("+" if x >= 0 else "") + dot(x)


def markdown(t1: pd.DataFrame, t2: pd.DataFrame, t3: pd.DataFrame) -> str:
    out = [
        "# Tables of the report",
        "",
        "Made by `scripts/06_make_report_tables.py` from `results/nested_cv/report_n38_cal/`.",
    ]
    out += ["", "## Table 1. RQ1: accuracy and compute", ""]
    out += [
        "Main columns: takes not used to pre-train CamFormer (n = 337). Last two columns: all takes (n = 1,271).",
        "Compute is the percentage of EgoVLP-only GMAC per take. The cascade uses the threshold gate with the cost "
        f"rule at β = {BETA}.",
        "",
        "| System | Acc [95% CI] | bAcc | Comp. % | Acc (all) | Comp. % (all) |",
        "|---|---|---|---|---|---|",
    ]
    for r in t1.itertuples():
        ci = "" if r.acc_ci_low == r.acc_ci_high else f" [{dot(r.acc_ci_low)}, {dot(r.acc_ci_high)}]"
        out.append(
            f"| {r.system} | {dot(r.acc)}{ci} | {dot(r.bacc)} | {r.compute_pct:.1f} | {dot(r.acc_all)} | "
            f"{r.compute_pct_all:.1f} |"
        )
    cascade = t1.iloc[-1]
    out += ["", f"Compute interval of the cascade: [{cascade.compute_ci_low:.1f}, {cascade.compute_ci_high:.1f}]%."]
    out += ["", "## Table 2. RQ2: gate ablation", ""]
    out += [
        "Takes not used to pre-train CamFormer (n = 337). Escalated and Acc at the cost rule with β = 0.05 (the class "
        "rule at its own share). Errors left: CamFormer errors per repeat left when the gate escalates the 30% of "
        "takes with the highest calibrated risk. ΔAUROC: against the row above, or class rule against threshold.",
        "",
        "| Gate | Features | AUROC [95% CI] | Escalated (%) | Acc | Errors left at 30% | ΔAUROC (p_Holm) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in t2.itertuples():
        delta = (
            "" if np.isnan(r.delta_auroc) else f"{signed(r.delta_auroc)} ({dot(r.p_holm, 3 if r.p_holm < 1 else 1)})"
        )
        if r.scorer == "lgbm_ABC":
            delta = "–"
        out.append(
            f"| {r.gate} | {r.features} | {dot(r.auroc)} [{dot(r.auroc_ci_low)}, {dot(r.auroc_ci_high)}] | "
            f"{r.escalated_pct:.1f} | {dot(r.acc)} | {r.errors_left_at_30pct:.1f} | {delta} |"
        )
    out += ["", "## Table 3. CamFormer's errors follow the activity type", ""]
    out += [
        "Takes not used to pre-train CamFormer, threshold gate with the cost rule at β = 0.05; escalation and errors "
        "left are per repeat.",
        "",
        "| Class | Takes | CamFormer acc | Escalated (%) | Errors left |",
        "|---|---|---|---|---|",
    ]
    for r in t3.itertuples():
        left = "0" if r.errors_left_per_repeat == 0 else f"{r.errors_left_per_repeat:.1f}"
        out.append(f"| {r._1} | {r.takes} | {r.camformer_acc:.2f} | {r.escalated_pct:.0f} | {left} |")
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=None, help="output folder (default: results/report/)")
    args = p.parse_args(argv)
    root, run = args.repo_root, args.repo_root / RUN
    out = args.out or root / OUT
    out.mkdir(parents=True, exist_ok=True)

    mc = metrics_with_cost(run)
    t1, t2, t3 = table1(mc), table2(run, mc), table3(run, root)
    t1.to_csv(out / "table1_rq1_accuracy_compute.csv", index=False, float_format="%.6f")
    t2.to_csv(out / "table2_rq2_gate_ablation.csv", index=False, float_format="%.6f")
    t3.to_csv(out / "table3_per_class.csv", index=False, float_format="%.6f")
    (out / "tables.md").write_text(markdown(t1, t2, t3), encoding="utf-8")
    print(f"Tables 1, 2 and 3 written to {out}")


if __name__ == "__main__":
    main()
