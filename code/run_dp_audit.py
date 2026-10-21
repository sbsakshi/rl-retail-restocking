"""
Exact planning results for every store layout (Section IV of the report).

    python run_dp_audit.py          # ~10 s, writes results/dp_audit.json

For each layout:
  * Value Iteration and Policy Iteration on the objective reward, with their
    agreement measured both literally and up to ties between equally good
    actions;
  * the optimal policy of each training reward, scored on the objective
    (does the reward change what is optimal?);
  * the best possible 200-step return (backward induction), which is the
    ceiling of the normalised score;
  * the effect of the discount factor;
  * the service-level / travel frontier as the price of one move varies;
  * a service-level upper bound that follows from travel times alone.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from mdp import StoreMDP
from retail_env import REWARD_PRESETS, OBJECTIVE_WEIGHTS
from evaluation import TIE_TOL

ROOT = Path(__file__).resolve().parent.parent
LAYOUTS = range(10)
T = 200

P = REWARD_PRESETS["proposal"]["weights"]
AUDIT_REWARDS = {
    "proposal": REWARD_PRESETS["proposal"],
    "proposal without closer bonus": {"weights": {k: v for k, v in P.items() if k != "closer"}},
    "objective + restock bonus": {"weights": {**OBJECTIVE_WEIGHTS, "restock_proactive": 10.0,
                                              "restock_reactive": 5.0}},
    "objective + closer bonus": {"weights": {**OBJECTIVE_WEIGHTS, "closer": 2.0}},
    "objective": REWARD_PRESETS["objective"],
    "pbrs": REWARD_PRESETS["pbrs"],
    "pbrs_stock": REWARD_PRESETS["pbrs_stock"],
}
GAMMAS = [0.8, 0.9, 0.95, 0.98, 0.99]
MOVE_COSTS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


def service_bound(mdp, T=T):
    """Upper bound on the expected service level from travel time alone.

    A unit delivered to shelf i costs at least d_i + 1 robot steps (half of a
    reload-travel-restock-return cycle carrying two units; splitting a load
    between shelves is never cheaper per unit). Expected sales on shelf i are
    at least p*T*SL_i, and sales cannot exceed initial stock plus deliveries.
    The robot also starts loaded and need not return after its last trip,
    which is allowed for with d_max + 2 extra steps.
    Maximise mean SL_i subject to sum_i (d_i+1) * max(0, p*T*SL_i - stock_max) <= T + slack.
    """
    d = np.array([abs(x - mdp.depot[0]) + abs(y - mdp.depot[1]) for x, y in mdp.shelves])
    c, n, pT = d + 1, len(d), mdp.demand_prob * T
    # variables: SL_1..SL_n, u_1..u_n with u_i >= pT*SL_i - stock_max, u_i >= 0
    obj = np.r_[-np.ones(n) / n, np.zeros(n)]
    A, b = [], []
    for i in range(n):
        row = np.zeros(2 * n); row[i] = pT; row[n + i] = -1
        A.append(row); b.append(mdp.stock_max)
    A.append(np.r_[np.zeros(n), c]); b.append(T + d.max() + 2)
    res = linprog(obj, A_ub=A, b_ub=b, bounds=[(0, 1)] * n + [(0, None)] * n)
    return float(-res.fun)


def audit(layout):
    mdp = StoreMDP(layout)
    R = mdp.objective()
    out = {"layout": layout, "shelves": mdp.shelves, "n_states": mdp.nS}

    t = time.time(); V_vi, Q_vi, pi_vi, sweeps = mdp.value_iteration(R); t_vi = time.time() - t
    t = time.time(); V_pi, Q_pi, pi_pi, iters = mdp.policy_iteration(R); t_pi = time.time() - t
    optimal = Q_vi >= Q_vi.max(axis=1, keepdims=True) - TIE_TOL
    out["vi_pi"] = {
        "vi_sweeps": sweeps, "vi_seconds": t_vi, "pi_iterations": iters, "pi_seconds": t_pi,
        "max_value_gap": float(np.abs(V_vi - V_pi).max()),
        "same_action": float(np.mean(pi_vi == pi_pi)),
        "pi_action_optimal_for_vi": float(np.mean(optimal[np.arange(mdp.nS), pi_pi])),
        "states_with_ties": int(np.sum(optimal.sum(axis=1) > 1)),
    }

    out["finite_horizon_optimum"] = float(mdp.finite_horizon_optimum(R, T))
    rand = mdp.episode_metrics(np.full((mdp.nS, mdp.nA), 1 / mdp.nA), T)
    out["random"] = {k: v for k, v in rand.items() if k != "occupancy"}
    span = out["finite_horizon_optimum"] - rand["objective"]
    score = lambda J: (J - rand["objective"]) / span

    out["rewards"] = {}
    for name, spec in AUDIT_REWARDS.items():
        _, Qr, pir, _ = mdp.value_iteration(mdp.reward(spec))
        e = mdp.episode_metrics(pir, T)
        row = {k: v for k, v in e.items() if k != "occupancy"}
        row["score"] = score(e["objective"])
        row["optimal_for_objective"] = float(np.mean(optimal[np.arange(mdp.nS), pir]))
        out["rewards"][name] = row

    out["gamma"] = {}
    for g in GAMMAS:
        _, _, pig, _ = mdp.value_iteration(R, gamma=g, tol=1e-9)
        e = mdp.episode_metrics(pig, T)
        out["gamma"][str(g)] = {"objective": e["objective"], "score": score(e["objective"]),
                                "service_level": e["service_level"], "distance": e["distance"]}

    out["frontier"] = []
    for w in MOVE_COSTS:
        _, _, piw, _ = mdp.value_iteration(mdp.reward({"weights": {"n_empty": -8.0, "moved": -w}}))
        e = mdp.episode_metrics(piw, T)
        out["frontier"].append({"move_cost": w, "service_level": e["service_level"],
                                "distance": e["distance"]})

    out["service_bound"] = service_bound(mdp)
    return out


if __name__ == "__main__":
    t0 = time.time()
    results = [audit(l) for l in LAYOUTS]
    path = ROOT / "results" / "dp_audit.json"
    path.write_text(json.dumps(results, indent=1))
    for r in results:
        o = r["rewards"]["objective"]
        print(f"layout {r['layout']} shelves {r['shelves']}: optimal SL {o['service_level']:.3f} "
              f"(bound {r['service_bound']:.3f}), travel {o['distance']:.0f}, "
              f"VI {r['vi_pi']['vi_sweeps']} sweeps / PI {r['vi_pi']['pi_iterations']} iterations, "
              f"agreement {r['vi_pi']['same_action']:.3f} literal, "
              f"{r['vi_pi']['pi_action_optimal_for_vi']:.3f} up to ties")
    print(f"done in {time.time() - t0:.1f}s -> {path}")
