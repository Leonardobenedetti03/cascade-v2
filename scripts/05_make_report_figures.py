"""Draw Figures 2 and 3 of the report from the files in results/.

Figure 1 of the report is a diagram of the cascade, drawn by hand in LaTeX, so it is not here.

Figure 2: the accuracy-compute frontier of the threshold gate with the cost rule, one point per price beta, for the
takes CamFormer was not trained on (a) and for all takes (b).
Figure 3: reliability of the threshold gate's calibrated risk on the takes CamFormer was not trained on.

It only reads results/nested_cv/report_n38_cal/ and writes to results/report/ the two figures (PDF and PNG) and the
numbers behind each of them (CSV). Nothing is rerun.

Run it from the repository root:
    python scripts/05_make_report_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RUN = Path("results/nested_cv/report_n38_cal")
OUT = Path("results/report")
BETAS = [0.20, 0.10, 0.05, 0.02, 0.01]  # left to right on the frontier
SUMMARY_BETA = 0.05
PANELS = [
    ("pretrain_unseen", "(a) Takes not used to pre-train CamFormer (n = 337)"),
    ("all", "(b) All takes (n = 1,271)"),
]
ORANGE, GREY = "#d95f02", "#9e9e9e"


def figure2_data(run: Path) -> pd.DataFrame:
    """One row per point of Figure 2. Compute is a percentage of running EgoVLP alone on every take."""
    m = pd.read_csv(run / "metrics.csv")
    c = pd.read_csv(run / "cost_vs_egovlp.csv")
    key = ["subset", "system", "rule", "param"]
    mc = m.merge(c[key + ["cost_frac_of_egovlp_only", "cost_frac_ci_low", "cost_frac_ci_high"]], on=key)
    rows = []
    for subset, _ in PANELS:
        s = mc[mc.subset == subset]
        for beta in BETAS:
            r = s[(s.system == "thr_maxprob") & (s.rule == "cost") & (s.param.round(4) == beta)].iloc[0]
            rows.append(
                {
                    "subset": subset,
                    "series": "cost_rule",
                    "beta": beta,
                    "compute_pct": 100 * r.cost_frac_of_egovlp_only,
                    "compute_ci_low": 100 * r.cost_frac_ci_low,
                    "compute_ci_high": 100 * r.cost_frac_ci_high,
                    "acc": r.acc,
                    "acc_ci_low": r.acc_ci_low,
                    "acc_ci_high": r.acc_ci_high,
                }
            )
        for system, series in [("oracle", "oracle"), ("always_L1", "egovlp_only"), ("always_L0", "camformer_only")]:
            r = s[(s.system == system) & (s.rule == "fixed")].iloc[0]
            rows.append(
                {"subset": subset, "series": series, "compute_pct": 100 * r.cost_frac_of_egovlp_only, "acc": r.acc}
            )
    return pd.DataFrame(rows)


def ranking_curve(run: Path) -> pd.DataFrame:
    """The grey line: escalate the r% of takes with the highest threshold score, for r = 0..100."""
    cv = pd.read_csv(run / "curves.csv")
    cv = cv[cv.scorer == "thr_maxprob"].copy()
    cv["compute_pct"] = 100 * cv.cost_frac_L1
    return cv[["subset", "r", "compute_pct", "acc"]]


def figure3_data(run: Path) -> pd.DataFrame:
    """The 10 equal-count bins of the calibrated risk, pooled over the 15 outer folds."""
    cal = pd.read_csv(run / "calibration.csv")
    q = cal[(cal.subset == "pretrain_unseen") & (cal.scorer == "thr_maxprob") & (cal.bin >= 0)]
    return q[["bin", "r_mean", "error_rate", "n"]].reset_index(drop=True)


def draw_figure2(points: pd.DataFrame, curve: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (subset, title) in zip(axes, PANELS):
        p = points[points.subset == subset]
        cv = curve[curve.subset == subset]
        ax.plot(cv.compute_pct, cv.acc, color=GREY, lw=1.0, label="threshold at every escalation share (ranking only)")
        rule = p[p.series == "cost_rule"]
        ax.errorbar(
            rule.compute_pct,
            rule.acc,
            xerr=[rule.compute_pct - rule.compute_ci_low, rule.compute_ci_high - rule.compute_pct],
            yerr=[rule.acc - rule.acc_ci_low, rule.acc_ci_high - rule.acc],
            color=ORANGE,
            marker="s",
            ms=3.5,
            lw=1.4,
            elinewidth=0.8,
            capsize=1.5,
            label="threshold gate, cost rule",
        )
        summary = rule[rule.beta == SUMMARY_BETA]
        ax.plot(
            summary.compute_pct,
            summary.acc,
            "s",
            ms=8,
            mfc="none",
            mec=ORANGE,
            mew=1.2,
            label="β = 0.05 (summary point)",
        )
        oracle = p[p.series == "oracle"]
        ax.plot(oracle.compute_pct, oracle.acc, "^", color="0.25", ms=5, label="oracle gate")
        egovlp = p[p.series == "egovlp_only"]
        ax.plot(egovlp.compute_pct, egovlp.acc, "D", color="black", ms=4, label="EgoVLP only")
        cam = p[p.series == "camformer_only"].iloc[0]
        ax.text(
            97,
            0.9615,
            f"CamFormer only: {cam.acc:.3f} at {cam.compute_pct:.1f}% (off scale)",
            ha="right",
            fontsize=6,
            color="0.35",
        )
        ax.set_xlim(-3, 103)
        ax.set_ylim(0.955, 1.005)
        ax.set_yticks([0.96, 0.98, 1.00])
        ax.set_xlabel("Compute (% of EgoVLP-only GMAC)")
        ax.set_title(title, fontsize=8)
        ax.grid(color="0.92", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # cost rule and summary point in the first column, then the reference line and points
    order = [
        "threshold gate, cost rule",
        "β = 0.05 (summary point)",
        "threshold at every escalation share (ranking only)",
        "oracle gate",
        "EgoVLP only",
    ]
    fig.legend([by_label[x] for x in order], order, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, out / "fig2_accuracy_compute")


def draw_figure3(bins: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    ax.plot([0, 0.6], [0, 0.6], ls=":", color="0.4", lw=1.0)
    ax.plot(bins.r_mean, bins.error_rate, "s-", color=ORANGE, ms=3.5, lw=1.2, label="threshold gate")
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(-0.02, 0.62)
    ax.set_xlabel("Predicted risk of a CamFormer error (bin mean)")
    ax.set_ylabel("Observed error rate")
    ax.grid(color="0.92", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, out / "fig3_reliability")


def save(fig, stem: Path) -> None:
    # No creation date in the files, so that drawing the same figure twice gives the same PDF.
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=None, help="output folder (default: results/report/)")
    args = p.parse_args(argv)
    run = args.repo_root / RUN
    out = args.out or args.repo_root / OUT
    out.mkdir(parents=True, exist_ok=True)

    points, curve, bins = figure2_data(run), ranking_curve(run), figure3_data(run)
    points.to_csv(out / "fig2_data.csv", index=False, float_format="%.6f")
    bins.to_csv(out / "fig3_data.csv", index=False, float_format="%.6f")
    draw_figure2(points, curve, out)
    draw_figure3(bins, out)
    print(f"Figures 2 and 3 written to {out}")


if __name__ == "__main__":
    main()
