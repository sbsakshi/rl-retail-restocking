"""
Build every table and figure used in the report from results/runs/*.json and
results/dp_audit.json.

    python analyze.py      # writes results/tables/*.csv|tex and results/figures/*.pdf|png

Statistics. Each cell aggregates 10 store layouts (one training run each).
Intervals are 95% t-intervals over layouts. Comparisons between training
rewards are paired by layout (same store, same seed), so they use the
paired t-test on per-layout differences.
"""

import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")      # files only; leave notebooks on their inline backend
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS, TABLES, FIGS = ROOT / "results/runs", ROOT / "results/tables", ROOT / "results/figures"

ALGOS = ["qlearning", "sarsa", "mc", "nstep_sarsa", "sarsa_lambda",
         "dqn", "double_dqn", "reinforce", "ppo"]
LABEL = {"qlearning": "Q-Learning", "sarsa": "SARSA", "mc": "MC control",
         "nstep_sarsa": "4-step Sarsa", "sarsa_lambda": r"SARSA($\lambda$)",
         "dqn": "DQN", "double_dqn": "Double DQN", "reinforce": "REINFORCE+b", "ppo": "PPO"}
PLAIN = {**LABEL, "sarsa_lambda": "SARSA(λ)"}
REWARDS = ["proposal", "objective", "pbrs", "pbrs_stock"]
RLABEL = {"proposal": "Proposal", "objective": "Objective", "pbrs": "PBRS-d",
          "pbrs_stock": "PBRS-ds"}
# categorical slots 1-4 of the reference palette; markers carry identity in print
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})
COL_W, PAGE_W = 3.45, 7.1          # IEEE column / page width in inches


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def load(tag=""):
    runs = collections.defaultdict(dict)
    for f in RUNS.glob("*.json"):
        r = json.loads(f.read_text())
        if r.get("tag", "") == tag:
            runs[(r["algo"], r["reward"])][r["layout"]] = r
    return runs


_mdps = {}


def enrich(runs):
    """Add metrics defined after the grid was run, recomputed exactly from each saved policy."""
    from mdp import StoreMDP
    for cell in runs.values():
        for layout, r in cell.items():
            if "stockout_duration" not in r["final"]:
                m = _mdps.setdefault(layout, StoreMDP(layout))
                r["final"]["stockout_duration"] = m.episode_metrics(np.array(r["policy"]))["stockout_duration"]
    return runs


def by_layout(cell, key):
    return np.array([key(cell[l]) for l in sorted(cell)])


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------- #
def table_scores(runs):
    """Final normalised score, algorithm x training reward, + paired tests vs Objective."""
    lines, csv = [], ["algo," + ",".join(f"{r}_mean,{r}_ci" for r in REWARDS)]
    tests = {}
    for a in ALGOS:
        cells, row = [], [a]
        base = by_layout(runs[(a, "objective")], lambda r: r["final"]["score"]) \
            if runs.get((a, "objective")) else None
        means = {r: ci95(by_layout(runs[(a, r)], lambda x: x["final"]["score"]))[0]
                 for r in REWARDS if runs.get((a, r))}
        best = max(means, key=means.get) if means else None
        for r in REWARDS:
            if not runs.get((a, r)):
                cells.append("--"); row += ["", ""]; continue
            s = by_layout(runs[(a, r)], lambda x: x["final"]["score"])
            m, h = ci95(s)
            mark = ""
            if r != "objective" and base is not None:
                p = stats.ttest_rel(s, base).pvalue
                tests[(a, r)] = (float(np.mean(s - base)), float(p))
                mark = r"$^{\ast}$" if p < 0.05 else ""
            txt = f"{m:.2f} $\\pm$ {h:.2f}{mark}"
            cells.append(f"\\textbf{{{txt}}}" if r == best else txt)
            row += [f"{m:.4f}", f"{h:.4f}"]
        lines.append(f"{LABEL[a]} & " + " & ".join(cells) + r" \\")
        csv.append(",".join(row))
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "scores.tex").write_text("\n".join(lines) + "\n")
    (TABLES / "scores.csv").write_text("\n".join(csv) + "\n")
    return tests


def table_metrics(runs, reward):
    """Operational metrics for every learner under one training reward, plus DP."""
    dp = json.loads((ROOT / "results/dp_audit.json").read_text())
    keys = ["score", "objective", "service_level", "distance", "stockout_duration",
            "proactive_share", "load_factor", "agree_opt_states"]
    fmt = {"score": "{:.2f}", "objective": "{:.0f}", "service_level": "{:.3f}",
           "distance": "{:.0f}", "stockout_duration": "{:.1f}", "proactive_share": "{:.2f}",
           "load_factor": "{:.2f}", "agree_opt_states": "{:.2f}"}
    lines, csv = [], ["algo," + ",".join(keys)]

    def emit(name, cols):
        lines.append(name + " & " + " & ".join(cols) + r" \\")

    from mdp import StoreMDP
    from evaluation import Reference
    dp_rows = []
    for d in dp:
        row = dict(d["rewards"]["objective"])
        ref = Reference(_mdps.setdefault(d["layout"], StoreMDP(d["layout"])))
        row["stockout_duration"] = ref.star["stockout_duration"]
        dp_rows.append(row)
    dp_cols, dp_csv = [], ["dp"]
    for k in keys:
        if k == "agree_opt_states":
            dp_cols.append("1.00"); dp_csv.append("1"); continue
        vals = [r[k] for r in dp_rows]
        m, h = ci95(vals)
        dp_cols.append(fmt[k].format(m)); dp_csv.append(f"{m:.4f}")
    emit(r"DP optimum $\pi^\ast$", dp_cols)
    csv.append(",".join(dp_csv))
    lines.append(r"\midrule")
    for a in ALGOS:
        cell = runs.get((a, reward))
        if not cell:
            continue
        cols, row = [], [a]
        for k in keys:
            m, h = ci95(by_layout(cell, lambda r: r["final"][k]))
            cols.append(fmt[k].format(m)); row.append(f"{m:.4f}")
        emit(LABEL[a], cols)
        csv.append(",".join(row))
    (TABLES / f"metrics_{reward}.tex").write_text("\n".join(lines) + "\n")
    (TABLES / f"metrics_{reward}.csv").write_text("\n".join(csv) + "\n")


def table_dp():
    dp = json.loads((ROOT / "results/dp_audit.json").read_text())
    lines = []
    for d in dp:
        o = d["rewards"]["objective"]
        shelves = ", ".join(f"({x},{y})" for x, y in d["shelves"])
        lines.append(f"{d['layout']} & {shelves} & {o['service_level']:.3f} & {d['service_bound']:.3f} & "
                     f"{o['distance']:.0f} & {d['random']['service_level']:.3f} & "
                     f"{d['vi_pi']['vi_sweeps']} & {d['vi_pi']['pi_iterations']} & "
                     f"{d['vi_pi']['same_action']:.3f} & {d['vi_pi']['states_with_ties']} \\\\")
    (TABLES / "dp_layouts.tex").write_text("\n".join(lines) + "\n")

    names = ["proposal", "proposal without closer bonus", "objective + restock bonus",
             "objective + closer bonus", "objective", "pbrs", "pbrs_stock"]
    shown = {"proposal": "Proposal", "proposal without closer bonus": "Proposal $-$ closer",
             "objective + restock bonus": "Objective $+$ restock", "objective + closer bonus":
             "Objective $+$ closer", "objective": "Objective", "pbrs": "PBRS-d", "pbrs_stock": "PBRS-ds"}
    lines = []
    for n in names:
        sc = [d["rewards"][n]["score"] for d in dp]
        dist = [d["rewards"][n]["distance"] for d in dp]
        sl = [d["rewards"][n]["service_level"] for d in dp]
        opt = [d["rewards"][n]["optimal_for_objective"] for d in dp]
        base_dist = [d["rewards"]["objective"]["distance"] for d in dp]
        extra = np.round(100 * (np.array(dist) / np.array(base_dist) - 1), 1) + 0.0   # no "-0.0"
        lines.append(f"{shown[n]} & {np.mean(sc):.4f} & {np.mean(sl):.4f} & {np.mean(dist):.1f} & "
                     f"${extra.mean():+.1f}$ [${extra.min():+.1f}$, ${extra.max():+.1f}$] & "
                     f"{100 * np.mean(opt):.1f} \\\\")
    (TABLES / "dp_rewards.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for g in ["0.8", "0.9", "0.95", "0.98", "0.99"]:
        sc = [d["gamma"][g]["score"] for d in dp]
        lines.append(f"{g} & {np.mean(sc):.4f} & {np.min(sc):.4f} \\\\")
    (TABLES / "dp_gamma.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------- #
def fig_frontier(runs, reward):
    """Exact service/travel frontier for three layouts, with learned policies placed on it."""
    dp = json.loads((ROOT / "results/dp_audit.json").read_text())
    show = [2, 0, 4]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.1), sharey=False)
    learners = ["sarsa_lambda", "nstep_sarsa", "qlearning", "dqn", "ppo"]
    for ax, l in zip(axes, show):
        fr = dp[l]["frontier"]
        ax.plot([f["distance"] for f in fr], [f["service_level"] for f in fr], "-",
                color=MUTED, lw=1.2, marker="o", ms=2.5, zorder=1, label="DP frontier")
        o = dp[l]["rewards"]["objective"]
        ax.scatter([o["distance"]], [o["service_level"]], marker="*", s=80, color=INK,
                   zorder=3, label=r"$\pi^\ast$ (move cost 0.1)")
        for i, a in enumerate(learners):
            r = runs.get((a, reward), {}).get(l)
            if r:
                ax.scatter(r["final"]["distance"], r["final"]["service_level"], s=26,
                           marker=MARKERS[i], color=COLORS[i], edgecolor="white",
                           linewidth=0.6, zorder=4, label=PLAIN[a])
        ax.axhline(dp[l]["service_bound"], color=MUTED, ls=":", lw=0.8)
        ax.set_title(f"Layout {l}: shelves {', '.join(map(str, map(tuple, dp[l]['shelves'])))}")
        ax.set_xlabel("travel (moves / episode)")
    axes[0].set_ylabel("service level")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    save(fig, f"frontier_{reward}")


def fig_curves(runs, reward):
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3), sharey=True)
    groups = [["qlearning", "sarsa", "mc", "nstep_sarsa", "sarsa_lambda"],
              ["dqn", "double_dqn", "reinforce", "ppo"]]
    for ax, group, title in zip(axes, groups, ["Tabular", "Function approximation"]):
        for i, a in enumerate(group):
            cell = runs.get((a, reward))
            if not cell:
                continue
            curves = np.array([[c["score"] for c in cell[l]["curve"]] for l in sorted(cell)])
            steps = np.array([c["steps"] for c in cell[sorted(cell)[0]]["curve"]]) / 1000
            m = curves.mean(0)
            h = stats.t.ppf(0.975, len(curves) - 1) * curves.std(0, ddof=1) / np.sqrt(len(curves))
            ax.plot(steps, m, color=COLORS[i], marker=MARKERS[i], markevery=6, ms=3.5,
                    label=PLAIN[a])
            ax.fill_between(steps, m - h, m + h, color=COLORS[i], alpha=0.15, lw=0)
        ax.axhline(1, color=MUTED, ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("environment steps (thousands)")
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("normalised score (exact)")
    save(fig, f"curves_{reward}")


def fig_reward_effect(runs):
    """Final score per algorithm, one marker per training reward (paired lines)."""
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.2))
    x = np.arange(len(ALGOS))
    width = 0.18
    for j, r in enumerate(REWARDS):
        ms, hs = [], []
        for a in ALGOS:
            cell = runs.get((a, r))
            m, h = ci95(by_layout(cell, lambda q: q["final"]["score"])) if cell else (np.nan, 0)
            ms.append(m); hs.append(h)
        ax.bar(x + (j - 1.5) * width, ms, width * 0.9, yerr=hs, color=COLORS[j],
               edgecolor="white", linewidth=0.6, label=RLABEL[r],
               error_kw={"elinewidth": 0.7, "ecolor": MUTED, "capsize": 0})
    wrap = {"mc": "MC\ncontrol", "nstep_sarsa": "4-step\nSarsa", "double_dqn": "Double\nDQN",
            "reinforce": "REINFORCE\n+ baseline"}
    ax.set_xticks(x, [wrap.get(a, PLAIN[a]) for a in ALGOS])
    ax.set_ylabel("normalised score")
    ax.axhline(1, color=MUTED, ls=":", lw=0.8)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              title="training reward", title_fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    save(fig, "reward_effect")


def fig_diagnostics(runs, reward):
    """Left: tabular value bias vs score (one point per run).
    Right: agreement with pi* vs score, one labelled point per learner (mean over layouts)."""
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3))
    tab = ["qlearning", "sarsa", "mc", "nstep_sarsa", "sarsa_lambda"]
    for i, a in enumerate(tab):
        cell = runs[(a, reward)]
        axes[0].scatter(by_layout(cell, lambda r: r["value_bias"]),
                        by_layout(cell, lambda r: r["final"]["score"]), s=14, marker=MARKERS[i],
                        color=COLORS[i], edgecolor="white", linewidth=0.4, label=PLAIN[a])
    axes[0].axvline(0, color=MUTED, lw=0.8, ls=":")
    axes[0].set_xlabel(r"value bias  $E_{d^{\pi^\ast}}[\max_a Q - V^\ast]$")
    axes[0].set_ylabel("normalised score")
    axes[0].set_title("Tabular: optimism left in the table")
    axes[0].legend(frameon=False, fontsize=6)
    for a in ALGOS:
        cell = runs[(a, reward)]
        ag = by_layout(cell, lambda r: r["final"]["agree_opt_states"]).mean()
        sc = by_layout(cell, lambda r: r["final"]["score"]).mean()
        deep_ = a in ("dqn", "double_dqn", "reinforce", "ppo")
        axes[1].scatter(ag, sc, s=22, marker="s" if deep_ else "o",
                        color=COLORS[0] if deep_ else COLORS[1], edgecolor="white", linewidth=0.5)
        axes[1].annotate(PLAIN[a], (ag, sc), xytext=(4, 2), textcoords="offset points",
                         fontsize=6, color=INK)
    axes[1].set_xlabel(r"P(optimal action) on states $\pi^\ast$ visits")
    axes[1].set_title("Agreement with $\\pi^\\ast$ (mean over layouts; square = network)")
    save(fig, f"diagnostics_{reward}")


def fig_layouts():
    """Store layouts with the optimal service level and the travel bound (column width)."""
    dp = json.loads((ROOT / "results/dp_audit.json").read_text())
    fig, axes = plt.subplots(2, 5, figsize=(COL_W, 1.75))
    for ax, d in zip(axes.ravel(), dp):
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("#c3c2b7"); spine.set_linewidth(0.5)
        for g in (0.5, 1.5, 2.5):
            ax.axhline(g, color="#e4e3df", lw=0.4); ax.axvline(g, color="#e4e3df", lw=0.4)
        ax.grid(False)
        ax.add_patch(plt.Rectangle((-0.42, -0.42), 0.84, 0.84, color=COLORS[0]))
        ax.text(0, 0, "D", ha="center", va="center", color="white", fontsize=5, weight="bold")
        for k, (x, y) in enumerate(d["shelves"]):
            ax.add_patch(plt.Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, color=COLORS[1]))
            ax.text(x, y, f"S{k+1}", ha="center", va="center", color="white", fontsize=4.5, weight="bold")
        o = d["rewards"]["objective"]
        ax.set_title(f"L{d['layout']}\n{o['service_level']:.2f} / {d['service_bound']:.2f}",
                     fontsize=5.5, pad=2, linespacing=1.1)
    fig.subplots_adjust(wspace=0.3, hspace=0.05)
    save(fig, "layouts")


def table_terms(runs):
    """Which proposal term helps: one term at a time, plus the potential-based forms."""
    cols = ["objective", "objective_restock", "objective_closer", "proposal", "pbrs", "pbrs_stock"]
    lines = []
    for a in ["qlearning", "sarsa", "nstep_sarsa", "sarsa_lambda", "dqn"]:
        base = by_layout(runs[(a, "objective")], lambda q: q["final"]["score"])
        cells = []
        for c in cols:
            sc = by_layout(runs[(a, c)], lambda q: q["final"]["score"])
            star = r"$^{\ast}$" if c != "objective" and stats.ttest_rel(sc, base).pvalue < 0.05 else ""
            cells.append(f"{sc.mean():.2f}{star}")
        lines.append(f"{LABEL[a]} & " + " & ".join(cells) + r" \\")
    (TABLES / "terms.tex").write_text("\n".join(lines) + "\n")


def table_init(runs):
    """Tabular learners on the Objective reward with different initial values; DQN at -150."""
    inits = [0, -50, -100, -150, -200, -300]
    tab = ["qlearning", "sarsa", "nstep_sarsa", "sarsa_lambda", "mc"]
    cells = {q: (runs if q == 0 else load(tag=f"init{q}")) for q in inits}
    lines = []
    for a in tab + ["dqn", "double_dqn"]:
        row = []
        best = max((ci95(by_layout(cells[q][(a, "objective")], lambda r: r["final"]["score"]))[0], q)
                   for q in inits if (a, "objective") in cells[q])
        for q in inits:
            if (a, "objective") not in cells[q]:
                row.append("--"); continue
            m, h = ci95(by_layout(cells[q][(a, "objective")], lambda r: r["final"]["score"]))
            txt = f"{m:.2f}"
            row.append(f"\\textbf{{{txt}}}" if q == best[1] else txt)
        prop = ci95(by_layout(runs[(a, "proposal")], lambda r: r["final"]["score"]))[0]
        lines.append(f"{LABEL[a]} & " + " & ".join(row) + f" & {prop:.2f} \\\\")
    (TABLES / "init.tex").write_text("\n".join(lines) + "\n")
    return cells


def fig_init(cells):
    inits = sorted(cells, reverse=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for i, a in enumerate(["qlearning", "sarsa", "nstep_sarsa", "sarsa_lambda", "mc"]):
        ms, hs = zip(*[ci95(by_layout(cells[q][(a, "objective")], lambda r: r["final"]["score"]))
                       for q in inits])
        ax.errorbar(inits, ms, yerr=hs, color=COLORS[i], marker=MARKERS[i], ms=3.5, capsize=0,
                    elinewidth=0.7, label=PLAIN[a])
    # reference values of V* (means over layouts, from the exact solution);
    # the x axis is inverted, so ha="right" puts a label on the visual left
    for x, txt, ha, dx in [(-137, "mean $V^\\ast$ on\nexplored states", "right", 5),
                           (-197, "smallest\n$V^\\ast$", "left", -5)]:
        ax.axvline(x, color=MUTED, ls=":", lw=0.8)
        ax.text(x + dx, 0.02, txt, fontsize=5.5, color=MUTED, ha=ha, va="bottom",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("initial value $Q_0$ (Objective reward)")
    ax.set_ylabel("normalised score")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "init")


def summary_numbers(runs, tests):
    """Plain-text dump of the numbers quoted in the report text."""
    out = []
    for (a, r), (d, p) in sorted(tests.items()):
        out.append(f"paired vs objective  {a:13s} {r:11s} mean diff {d:+.3f}  p={p:.4f}")
    for a in ALGOS:
        for r in REWARDS:
            cell = runs.get((a, r))
            if not cell:
                continue
            sc = by_layout(cell, lambda q: q["final"]["score"])
            auc = by_layout(cell, lambda q: np.mean([c["score"] for c in q["curve"]]))
            reached = [next((c["steps"] for c in q["curve"] if c["score"] >= 0.5), None)
                       for q in (cell[l] for l in sorted(cell))]
            n_reached = sum(x is not None for x in reached)
            med = np.median([x for x in reached if x is not None]) if n_reached else float("nan")
            extra = ""
            if "stochastic" in cell[sorted(cell)[0]]:
                st = by_layout(cell, lambda q: q["stochastic"]["score"])
                extra = f"  stochastic {ci95(st)[0]:.3f}±{ci95(st)[1]:.3f}"
            if "value_bias" in cell[sorted(cell)[0]]:
                vb = by_layout(cell, lambda q: q["value_bias"])
                cov = by_layout(cell, lambda q: q["coverage"])
                extra += f"  bias {vb.mean():+.1f}  coverage {cov.mean():.2f}"
            secs = by_layout(cell, lambda q: q["train_seconds"]).mean()
            out.append(f"{a:13s} {r:11s} score {ci95(sc)[0]:.3f}±{ci95(sc)[1]:.3f}  "
                       f"AUC {auc.mean():.3f}  reach0.5 {n_reached}/10 median {med:.0f} steps  "
                       f"train {secs:.0f}s{extra}")
    (TABLES / "summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    runs = enrich(load())
    tests = table_scores(runs)
    table_dp()
    for r in REWARDS:
        table_metrics(runs, r)
    table_terms(runs)
    fig_init(table_init(runs))
    fig_layouts()
    fig_reward_effect(runs)
    for r in REWARDS:
        fig_curves(runs, r)
        fig_diagnostics(runs, r)
        fig_frontier(runs, r)
    summary_numbers(runs, tests)
