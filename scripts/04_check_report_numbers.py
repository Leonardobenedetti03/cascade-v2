"""Check the numbers in the report against the files in results/.

For every number in the report that comes from an experiment, this script recomputes it from the result files and
prints it next to the value in the paper, rounded the same way. It only reads files: nothing is rerun.

It reads two folders:
    results/nested_cv/report_n38_cal/                 the nested cross-validation run used in the report
    results/features/official_v2/heads/camformer/     labels, sessions, and which takes CamFormer was trained on

Run it from the repository root:
    python scripts/04_check_report_numbers.py

It exits with 0 if every number matches and with 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

RUN = Path("results/nested_cv/report_n38_cal")
PRED = Path("results/features/official_v2/heads/camformer")
U = "pretrain_unseen"  # takes CamFormer was not trained on: the main subset of the report
CLASSES = [
    "Basketball",
    "Dance",
    "Soccer",
    "Cooking",
    "Health",
    "Bike repair",
]  # same order as the heads' outputs

Row = Tuple[str, str, str, str, str]


class Checker:
    """Keeps one row per number checked."""

    def __init__(self) -> None:
        self.rows: List[Row] = []

    def num(self, where: str, what: str, reported: float, value: float, ndig: int = 3) -> None:
        """Two numbers match if they round to the same value at the precision used in the report."""
        ok = round(float(value), ndig) == round(float(reported), ndig)
        self.rows.append(
            (
                where,
                what,
                f"{reported}",
                f"{float(value):.{ndig}f}",
                "OK" if ok else "DIFF",
            )
        )

    def flag(self, where: str, what: str, reported: str, value: str, ok: bool) -> None:
        self.rows.append((where, what, reported, value, "OK" if ok else "DIFF"))


def one(df: pd.DataFrame, where: str) -> pd.Series:
    """Return the one row that matches. Zero or several rows mean the check itself is wrong."""
    if len(df) != 1:
        raise LookupError(f"{where}: expected 1 row, found {len(df)}")
    return df.iloc[0]


def check(root: Path) -> Checker:
    run = root / RUN
    m = pd.read_csv(run / "metrics.csv")
    c = pd.read_csv(run / "cost_vs_egovlp.csv")
    key = ["subset", "system", "rule", "param"]
    cost_cols = [
        "cost_frac_of_egovlp_only",
        "cost_frac_ci_low",
        "cost_frac_ci_high",
        "gmacs_per_take_cascade",
        "gmacs_per_take_egovlp_only",
        "speedup",
    ]
    mc = m.merge(c[key + cost_cols], on=key, how="left")
    auc = pd.read_csv(run / "auc.csv")
    cmp_ = pd.read_csv(run / "comparisons.csv")
    proc = pd.read_csv(run / "procedural.csv")
    curves = pd.read_csv(run / "curves.csv")
    cal = pd.read_csv(run / "calibration.csv")
    imp = pd.read_csv(run / "importance.csv")
    takes = pd.read_csv(run / "takes.csv.gz")
    models = pd.read_csv(run / "models.csv").set_index("subset")
    k = Checker()

    def row(subset: str, system: str, rule: str, param: float) -> pd.Series:
        q = mc[(mc.subset == subset) & (mc.system == system) & (mc.rule == rule) & np.isclose(mc.param, param)]
        return one(q, f"metrics {subset}/{system}/{rule}/{param}")

    # Table 1
    # each row: name, system, rule, param, acc, 95% CI, bAcc, compute %, acc on all takes, compute % on all takes
    t1 = [
        (
            "CamFormer only",
            "always_L0",
            "fixed",
            0.0,
            0.882,
            (0.826, 0.922),
            0.803,
            0.1,
            0.924,
            0.1,
        ),
        ("EgoVLP only", "always_L1", "fixed", 0.0, 1.000, None, 1.000, 100, 0.999, 100),
        ("Oracle gate", "oracle", "fixed", 0.0, 1.000, None, 1.000, 11.9, 1.000, 7.7),
        (
            "Class rule",
            "class_rule",
            "natural",
            0.0,
            0.985,
            (0.963, 0.999),
            0.989,
            29.6,
            0.985,
            22.8,
        ),
        (
            "Cascade (threshold)",
            "thr_maxprob",
            "cost",
            0.05,
            0.997,
            (0.989, 1.00),
            0.992,
            42.4,
            0.998,
            38.7,
        ),
    ]
    for name, system, rule, p, acc, ci, bacc, comp, acc_all, comp_all in t1:
        a, b = row(U, system, rule, p), row("all", system, rule, p)
        k.num("Tab.1", f"{name}: accuracy", acc, a.acc)
        if ci:
            k.num("Tab.1", f"{name}: CI low", ci[0], a.acc_ci_low)
            k.num(
                "Tab.1",
                f"{name}: CI high",
                ci[1],
                a.acc_ci_high,
                2 if ci[1] == 1.0 else 3,
            )
        k.num("Tab.1", f"{name}: balanced accuracy", bacc, a.bacc)
        k.num(
            "Tab.1",
            f"{name}: compute % of EgoVLP only",
            comp,
            100 * a.cost_frac_of_egovlp_only,
            1,
        )
        k.num("Tab.1", f"{name}: accuracy, all takes", acc_all, b.acc)
        k.num(
            "Tab.1",
            f"{name}: compute %, all takes",
            comp_all,
            100 * b.cost_frac_of_egovlp_only,
            1,
        )
    a = row(U, "thr_maxprob", "cost", 0.05)
    k.num("Tab.1", "cascade compute CI low %", 32.5, 100 * a.cost_frac_ci_low, 1)
    k.num("Tab.1", "cascade compute CI high %", 55.5, 100 * a.cost_frac_ci_high, 1)

    # Section 5.2
    k.num("5.2", "escalated % at beta = 0.05", 42.2, 100 * a.rho, 1)
    k.num("5.2", "GMACs per take, cascade", 195.8, a.gmacs_per_take_cascade, 1)
    k.num("5.2", "GMACs per take, EgoVLP only", 462.2, a.gmacs_per_take_egovlp_only, 1)
    k.num("5.2", "speed-up (2.4x)", 2.4, a.speedup, 1)
    gap = (a.acc - models.loc[U, "acc_L0"]) / (models.loc[U, "acc_L1"] - models.loc[U, "acc_L0"])
    k.num("5.2", "share of the accuracy gap closed (97%)", 97, 100 * gap, 0)
    k.num(
        "5.2",
        "speed-up, all takes (2.6x)",
        2.6,
        row("all", "thr_maxprob", "cost", 0.05).speedup,
        1,
    )
    for beta, acc, comp in [(0.01, 1.000, 61.3), (0.20, 0.985, 25.5)]:
        q = row(U, "thr_maxprob", "cost", beta)
        k.num("5.2", f"beta = {beta}: accuracy", acc, q.acc)
        k.num(
            "5.2",
            f"beta = {beta}: compute %",
            comp,
            100 * q.cost_frac_of_egovlp_only,
            1,
        )
    q = row(U, "thr_maxprob", "cost_hat", 0.05)
    k.num("5.2", "r-hat rule: escalated %", 36.2, 100 * q.rho, 1)
    k.num("5.2", "r-hat rule: accuracy", 0.993, q.acc)
    k.num("5.2", "r-hat rule: errors left per repeat", 2.3, q.errors_left, 1)
    k.num("5.2", "r-up rule: errors left per repeat", 1.0, a.errors_left, 1)

    # CamFormer and EgoVLP on their own (introduction, sections 3 and 4)
    k.num(
        "3.4/4",
        "CamFormer accuracy on its pre-training takes",
        0.940,
        models.loc["pretrain_seen", "acc_L0"],
    )
    k.num(
        "3.4/4",
        "CamFormer accuracy on the primary takes",
        0.882,
        models.loc[U, "acc_L0"],
    )
    k.num(
        "3.2",
        "EgoVLP errors over 3,813 take-repeats",
        4,
        models.loc["all", "errors_L1"] * 3,
        0,
    )
    k.num(
        "Intro",
        "CamFormer accuracy, all takes (92%)",
        0.92,
        models.loc["all", "acc_L0"],
        2,
    )
    k.num(
        "Intro",
        "CamFormer accuracy, primary takes (88%)",
        0.88,
        models.loc[U, "acc_L0"],
        2,
    )
    k.num(
        "Tab.2",
        "CamFormer errors per repeat (39.7)",
        39.7,
        models.loc[U, "errors_L0"],
        1,
    )

    # Table 2
    # each row: gate, AUROC, 95% CI, % sent to EgoVLP, accuracy, errors left when 30% of takes go to EgoVLP
    t2 = [
        ("thr_maxprob", 0.914, (0.844, 0.957), 42.2, 0.997, 3.3),
        ("logreg_A", 0.915, (0.845, 0.957), 45.2, 0.997, 4.0),
        ("logreg_AC", 0.911, (0.845, 0.958), 44.3, 0.996, 4.0),
        ("logreg_ABC", 0.908, (0.837, 0.960), 42.9, 0.995, 4.3),
        ("lgbm_ABC", 0.902, (0.836, 0.953), 56.8, 0.996, 3.3),
        ("class_rule", 0.784, (0.677, 0.875), 29.5, 0.985, 5.0),
    ]
    n_u = int(models.loc[U, "n_takes"])
    for g, au, ci, esc, acc, e30 in t2:
        q = one(auc[(auc.subset == U) & (auc.scorer == g)], f"auc {g}")
        k.num("Tab.2", f"{g}: AUROC", au, q.auroc)
        k.num("Tab.2", f"{g}: AUROC CI low", ci[0], q.auroc_ci_low)
        k.num("Tab.2", f"{g}: AUROC CI high", ci[1], q.auroc_ci_high)
        rule, p = ("natural", 0.0) if g == "class_rule" else ("cost", 0.05)
        q = row(U, g, rule, p)
        k.num("Tab.2", f"{g}: escalated %", esc, 100 * q.rho, 1)
        k.num("Tab.2", f"{g}: accuracy", acc, q.acc)
        cv = curves[(curves.subset == U) & (curves.scorer == g)]
        q30 = cv.loc[(cv.r - 30).abs().idxmin()]  # the point of the curve closest to 30% sent to EgoVLP
        k.num("Tab.2", f"{g}: errors left at 30% escalated", e30, (1 - q30.acc) * n_u, 1)

    def comparison(a_: str, b_: str, metric: str) -> pd.Series:
        q = cmp_[(cmp_.subset == U) & (cmp_.a == a_) & (cmp_.b == b_) & (cmp_.metric == metric)]
        return one(q, f"comparison {a_} - {b_} on {metric}")

    for a_, b_, d, ph in [
        ("logreg_A", "thr_maxprob", 0.001, 1.0),
        ("logreg_AC", "logreg_A", -0.004, 1.0),
        ("logreg_ABC", "logreg_AC", -0.003, 1.0),
        ("thr_maxprob", "class_rule", 0.130, 0.016),
    ]:
        q = comparison(a_, b_, "auroc")
        k.num("Tab.2", f"dAUROC {a_} - {b_}", d, q["diff"])
        k.num("Tab.2", f"p (Holm) {a_} - {b_}", ph, q.p_holm, 3 if ph < 1 else 2)

    # Section 5.3
    for a_, b_, d, ph in [
        ("logreg_A", "thr_maxprob", 3.0, 0.006),
        ("logreg_AC", "logreg_A", -0.9, 0.69),
        ("thr_maxprob", "class_rule", 12.8, None),
    ]:
        q = comparison(a_, b_, "rho_at_primary")
        k.num("5.3", f"escalation difference {a_} - {b_} (points)", d, 100 * q["diff"], 1)
        if ph is not None:
            k.num(
                "5.3",
                f"p (Holm) escalation {a_} - {b_}",
                ph,
                q.p_holm,
                3 if ph < 0.1 else 2,
            )
        else:
            k.flag(
                "5.3",
                f"p (Holm) escalation {a_} - {b_}",
                "<0.001",
                f"{q.p_holm:.4f}",
                q.p_holm < 0.001,
            )
    q = comparison("thr_maxprob", "class_rule", "acc_at_primary")
    k.num(
        "5.3",
        "accuracy difference threshold - class rule (points)",
        1.2,
        100 * q["diff"],
        1,
    )
    k.num("5.3", "p (Holm) accuracy threshold - class rule", 0.83, q.p_holm, 2)
    pp = one(
        proc[
            (proc.subset == U) & (proc.system == "thr_maxprob") & (proc.rule == "cost") & np.isclose(proc.param, 0.05)
        ],
        "procedural",
    )
    k.num("5.3", "procedural predictions per repeat (99)", 99, pp.n_pred_procedural, 0)
    k.num("5.3", "procedural kept on CamFormer (6.3)", 6.3, pp.kept_on_L0, 1)
    k.num("5.3", "procedural kept and wrong (1.0)", 1.0, pp.kept_errors, 1)
    k.num("5.3", "procedural kept share (6%)", 6, 100 * pp.frac_kept, 0)

    # Section 5.4
    lg = imp[imp.scorer == "logreg_AC"]
    for f in ["mot_energy", "mot_angvel_std"]:
        w = lg[lg.feature == f]
        n_neg = int((w.importance < 0).sum())
        k.flag(
            "5.4",
            f"logreg_AC weight of {f} < 0 in every fold",
            "15/15",
            f"{n_neg}/{len(w)}",
            n_neg == len(w) == 15,
        )
    gb = imp[imp.scorer == "lgbm_ABC"].copy()
    gb["share"] = gb.importance / gb.groupby(["repeat", "fold"]).importance.transform("sum")
    mot = gb[gb.feature.str.startswith("mot_")].groupby(["repeat", "fold"]).share.sum().mean()
    k.num(
        "5.4",
        "LightGBM split-gain share of the motion features (32%)",
        32,
        100 * mot,
        0,
    )

    # Section 5.5: calibration
    cq = one(
        cal[(cal.subset == U) & (cal.scorer == "thr_maxprob") & (cal.bin == -1)],
        "calibration",
    )
    k.num("5.5", "mean predicted risk (10.5%)", 10.5, 100 * cq.r_mean, 1)
    k.num("5.5", "observed error rate (11.8%)", 11.8, 100 * cq.error_rate, 1)
    k.num("5.5", "expected calibration error (0.016)", 0.016, cq.ece)

    # Section 4 and Table 3 (these need the labels from the feature run)
    pred = pd.concat(
        [
            pd.read_csv(root / PRED / "predictions.csv"),
            pd.read_csv(root / PRED / "sealed/predictions.csv"),
        ]
    )
    lab = pred.drop_duplicates("take_name").set_index("take_name")
    t = takes.join(lab[["label", "camformer_pretrain", "session"]], on="take_name")
    t["y"] = t.label.map({c_: i for i, c_ in enumerate(CLASSES)})
    one_per_take = t.drop_duplicates("take_name")
    n_t, n_s = t.take_name.nunique(), t.session.nunique()
    k.flag(
        "4",
        "takes / sessions",
        "1271 / 294",
        f"{n_t} / {n_s}",
        (n_t, n_s) == (1271, 294),
    )
    counts = one_per_take.label.value_counts()
    for c_, n in [
        ("Basketball", 468),
        ("Dance", 452),
        ("Cooking", 123),
        ("Bike repair", 81),
        ("Health", 77),
        ("Soccer", 70),
    ]:
        k.num("4", f"takes of {c_}", n, counts.get(c_, 0), 0)
    tu = t[t.camformer_pretrain != "train"].copy()  # the main subset: takes CamFormer was not trained on
    n_tu, n_su = tu.take_name.nunique(), tu.session.nunique()
    k.flag(
        "4",
        "primary takes / sessions",
        "337 / 108",
        f"{n_tu} / {n_su}",
        (n_tu, n_su) == (337, 108),
    )
    n_val = int((tu.drop_duplicates("take_name").camformer_pretrain == "val").sum())
    k.num("4", "primary takes in CamFormer's official val split (271)", 271, n_val, 0)

    tu["esc"] = tu.rup__thr_maxprob >= 0.05  # the cost rule at beta = 0.05, on the conservative risk
    tu["c0"] = tu.pred0 == tu.y
    tu["correct"] = np.where(tu.esc, tu.pred1 == tu.y, tu.c0)
    # each row: class, takes, CamFormer accuracy, % sent to EgoVLP, errors left per repeat
    t3 = [
        ("Dance", 116, 1.00, 19, 0),
        ("Basketball", 112, 0.93, 21, 0),
        ("Soccer", 13, 0.95, 56, 0),
        ("Cooking", 39, 0.73, 91, 0),
        ("Health", 35, 0.74, 97, 0),
        ("Bike repair", 22, 0.47, 91, 1.0),
    ]
    for c_, n, acc0, e, left in t3:
        g = tu[tu.label == c_]
        k.num("Tab.3", f"{c_}: takes", n, g.take_name.nunique(), 0)
        k.num("Tab.3", f"{c_}: CamFormer accuracy", acc0, g.c0.mean(), 2)
        k.num("Tab.3", f"{c_}: escalated %", e, 100 * g.esc.mean(), 0)
        k.num("Tab.3", f"{c_}: errors left per repeat", left, (~g.correct).sum() / 3, 1)

    # Section 5.3: the errors the cascade still makes
    res = tu[~tu.correct]
    names = sorted(res.take_name.unique())
    k.flag(
        "5.3",
        "residual errors: takes",
        "georgiatech_bike_06_2, _6",
        ", ".join(names),
        names == ["georgiatech_bike_06_2", "georgiatech_bike_06_6"],
    )
    wrong_as = sorted({CLASSES[int(i)] for i in res.pred0})
    k.flag(
        "5.3",
        "residual errors: Bike repair predicted as",
        "Cooking",
        ", ".join(wrong_as),
        set(res.label) == {"Bike repair"} and wrong_as == ["Cooking"],
    )
    k.flag(
        "5.3",
        "residual errors: CamFormer p_max",
        "0.978-0.995",
        f"{res.p_max.min():.3f}-{res.p_max.max():.3f}",
        round(res.p_max.min(), 3) == 0.978 and round(res.p_max.max(), 3) == 0.995,
    )
    # Section 5.5: the risk guarantee, fold by fold
    # In each fold, look at the main-subset takes that stay on CamFormer (score at or below that fold's threshold).
    # The guarantee holds if CamFormer is wrong on at most a fraction alpha of them. A fold that keeps no take
    # holds by default.
    thr = pd.read_csv(run / "thresholds.csv")
    risk = thr[(thr.scorer == "thr_maxprob") & (thr.rule == "risk")]
    for alpha, n_rep in [(0.02, 15), (0.05, 13), (0.10, 12)]:
        n_ok, n_folds, n_kept = 0, 0, 0
        for (rep, fold), g in tu.groupby(["repeat", "fold"]):
            lam = one(
                risk[(risk.repeat == rep) & (risk.fold == fold) & np.isclose(risk.param, alpha)],
                "risk",
            ).threshold
            kept = g[g.s__thr_maxprob <= lam]
            n_ok += int(len(kept) == 0 or (~kept.c0).mean() <= alpha)
            n_folds += 1
            n_kept += len(kept)
        k.flag(
            "5.5",
            f"risk guarantee holds, alpha = {alpha}",
            f"{n_rep}/15",
            f"{n_ok}/{n_folds}",
            (n_ok, n_folds) == (n_rep, 15),
        )
        if alpha == 0.02:
            k.flag(
                "5.5",
                "alpha = 0.02 escalates every take",
                "0 kept",
                f"{n_kept} kept",
                n_kept == 0,
            )
    k.num(
        "5.5",
        "P(12 or fewer of 15 folds hold) at the nominal 90% (18%)",
        18,
        100 * binom.cdf(12, 15, 0.9),
        0,
    )
    of = pd.read_csv(run / "outer_folds.csv").join(lab["camformer_pretrain"], on="take_name")
    seen = [
        (of[(of.repeat == r_) & (of.fold != f_)].camformer_pretrain == "train").mean()
        for r_, f_ in of[["repeat", "fold"]].drop_duplicates().itertuples(index=False)
    ]
    k.num(
        "5.5",
        "outer-training takes seen in CamFormer's pre-training (73%)",
        73,
        100 * np.mean(seen),
        0,
    )

    # Other numbers in the text
    c0, c1 = (
        a.gmacs_per_take_cascade - a.rho * a.gmacs_per_take_egovlp_only,
        a.gmacs_per_take_egovlp_only,
    )
    k.num("Abstract", "CamFormer about 760 times cheaper than EgoVLP", 760, c1 / c0, 0)
    k.num("3.5", "C0 / 5: CamFormer GMACs per window", 0.1216, c0 / 5, 4)
    k.num("3.5", "C1 / 5: EgoVLP GMACs per window", 92.43, c1 / 5, 2)
    k.num(
        "4",
        "share of takes in the two largest classes (72%)",
        72,
        100 * (468 + 452) / n_t,
        0,
    )
    k.num(
        "Limitations",
        "break-even cost of pose estimation (about 266 GMACs per take)",
        266,
        (1 - a.rho) * c1 - c0,
        0,
    )
    return k


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args(argv)
    k = check(args.repo_root)
    out = pd.DataFrame(k.rows, columns=["section", "quantity", "report", "files", "status"])
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.max_colwidth", 70):
        print(out.to_string(index=False))
    n_ok = int((out.status == "OK").sum())
    print(f"\n{n_ok} / {len(out)} numbers match the report")
    return 0 if n_ok == len(out) else 1


if __name__ == "__main__":
    sys.exit(main())
