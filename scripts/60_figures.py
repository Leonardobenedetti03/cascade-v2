"""Stadio 60: figure del report dai risultati finali (solo lettura, nessun calcolo nuovo).

Legge results/scenario_v1/ e scrive in results/scenario_v1/figures/ (PDF per LaTeX, PNG per slide e poster):
  fig1_accuracy_cost.{pdf,png}   accuratezza-costo su val e test con oracolo, scelta a caso e regola procedurale
  fig2_escalation_by_class.{pdf,png}  quota di take mandata a EgoVLP per classe (gate lgbm_ABC)
  fig3_motion_l0.{pdf,png}       accuratezza di CamFormer per quintile di mot_energy (asse x a numeri interi)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

CLASSES = ["Basketball", "Dance", "Soccer", "Cooking", "Health", "Bike repair"]
GATE = "lgbm_ABC"
CURVE_MODELS = {
    GATE: ("Gate appreso (lgbm_ABC)", "#1f4e9a", "-"),
    "thr_maxprob": ("Solo confidenza (thr_maxprob)", "#e07b00", "--"),
    "random": ("Scelta a caso", "#8c8c8c", ":"),
}
POINTS = {
    "always_L0": ("Sempre CamFormer", "o", "#2a9d8f"),
    "always_L1": ("Sempre EgoVLP", "s", "#000000"),
    "oracle": ("Oracolo", "*", "#c1121f"),
    "rule_procedural": ('Regola "procedurale → EgoVLP"', "D", "#6a4c93"),
}
SPLIT_NAME = {"val": "Validazione (143 take)", "test": "Test (142 take)"}


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"scritta {name}")


def fig_accuracy_cost(root: Path, out: Path) -> None:
    curves = pd.read_csv(root / "50_gate/seed0_hz20/main/curves.csv")
    # punti di lavoro (regola procedurale compresa) dalla tabella dello stadio 53
    op = pd.read_csv(root / "53_analysis/seed0_hz20/main/op_all.csv")
    op = op[op["subset"] == "all"]
    splits = [s for s in ("val", "test") if s in set(curves["split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.2), sharey=True)
    axes = [axes] if len(splits) == 1 else list(axes)
    for ax, split in zip(axes, splits):
        c = curves[curves["split"] == split]
        full = c.loc[c["model"] == "always_L1", "cost"].max()
        for model, (label, color, ls) in CURVE_MODELS.items():
            m = c[c["model"] == model]
            if m.empty:
                continue
            # più semi (scelta a caso): media per passo r
            m = m.groupby("r", as_index=False)[["cost", "acc"]].mean().sort_values("cost")
            ax.plot(100 * m["cost"] / full, 100 * m["acc"], ls, color=color, lw=1.8, label=label)
        o = op[(op["split"] == split)]
        for model, (label, marker, color) in POINTS.items():
            p = o[o["model"] == model].head(1)
            if p.empty:
                continue
            ax.scatter(
                100 * p["cost_frac_L1"],
                100 * p["acc"],
                marker=marker,
                s=90 if marker == "*" else 45,
                color=color,
                zorder=5,
                label=label,
                edgecolor="white",
                linewidth=0.6,
            )
        g = o[o["model"] == GATE].head(1)
        if not g.empty:
            x, y = 100 * g["cost_frac_L1"].iloc[0], 100 * g["acc"].iloc[0]
            ax.scatter(x, y, s=110, facecolor="none", edgecolor=CURVE_MODELS[GATE][1], lw=1.8, zorder=6)
            ax.annotate(
                f"punto di lavoro\n{y:.1f}% acc, costo {x:.1f}%",
                (x, y),
                xytext=(48, 92.3),
                fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8},
            )
        ax.set_title(SPLIT_NAME.get(split, split))
        ax.set_xlabel('Costo (% del calcolo di "sempre EgoVLP")')
        ax.set_xlim(-2, 102)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuratezza (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.16))
    save(fig, out, "fig1_accuracy_cost")


def fig_escalation_by_class(root: Path, out: Path) -> None:
    e = pd.read_csv(root / "50_gate/seed0_hz20/main/escalation_by_class.csv")
    e = e[e["model"] == GATE]
    splits = [s for s in ("val", "test") if s in set(e["split"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    width = 0.8 / len(splits)
    for i, split in enumerate(splits):
        s = e[e["split"] == split].set_index("class").reindex(CLASSES)
        x = [k + (i - (len(splits) - 1) / 2) * width for k in range(len(CLASSES))]
        bars = ax.bar(
            x,
            100 * s["escalation_rate"],
            width,
            label=SPLIT_NAME.get(split, split),
            color="#9db4d9" if split == "val" else "#1f4e9a",
        )
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
    proc = e.drop_duplicates("class").set_index("class")["procedural"].reindex(CLASSES)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(
        [f"{c}\n({'procedurale' if str(p) == 'True' else 'fisica'})" for c, p in proc.items()], fontsize=8
    )
    ax.set_ylabel("Take mandate a EgoVLP (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Quando il gate chiama EgoVLP, per classe (lgbm_ABC)")
    save(fig, out, "fig2_escalation_by_class")


def fig_motion(root: Path, out: Path, min_n: int) -> None:
    m = pd.read_csv(root / "53_analysis/seed0_hz20/main/motion_l0.csv")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for cls in ["all", *CLASSES]:
        d = m[(m["class"] == cls) & (m["n"] >= min_n)].sort_values("quintile")
        if len(d) < 2:
            continue
        if cls == "all":
            ax.plot(d["quintile"], 100 * d["acc_L0"], "-o", color="black", lw=2.4, label="Tutte le classi", zorder=5)
        else:
            ax.plot(d["quintile"], 100 * d["acc_L0"], "-o", lw=1.1, ms=3.5, alpha=0.8, label=cls)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    n_q = int(m.loc[m["class"] == "all", "n"].median())
    ax.set_xlabel(f"Quintile di energia del moto (1 = camera più ferma, 5 = più mossa)\n(~{n_q} take ciascuno)")
    ax.set_ylabel("Accuratezza di CamFormer (%)")
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=7,
        frameon=False,
        ncol=2,
        loc="lower right",
        title=f"classi: solo quintili con ≥ {min_n} take",
        title_fontsize=7,
    )
    ax.margins(y=0.08)
    ax.set_title("CamFormer e moto della camera (train, predizioni fuori campione)")
    save(fig, out, "fig3_motion_l0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results/scenario_v1"))
    ap.add_argument("--min-n", type=int, default=10, help="punti per classe con meno take non disegnati (fig. 3)")
    a = ap.parse_args()
    out = a.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_accuracy_cost(a.results, out)
    fig_escalation_by_class(a.results, out)
    fig_motion(a.results, out, a.min_n)


if __name__ == "__main__":
    main()
