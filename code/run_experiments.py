"""
Train every learner under every training reward on every store layout.

    python run_experiments.py                     # main grid: 9 learners x 3 rewards x 10 layouts
    python run_experiments.py --algos ppo --rewards pbrs --episodes 3000 --tag long
    python run_experiments.py --workers 4         # fewer parallel processes
    python run_experiments.py --algos qlearning sarsa --rewards objective --q-init -100 --tag init-100

One JSON file per run goes to results/runs/. A run that already has a file
is skipped, so an interrupted grid can be resumed by re-running the command.
analyze.py turns the files into the tables and figures of the report.
"""

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"

TABULAR = ["qlearning", "sarsa", "mc", "nstep_sarsa", "sarsa_lambda"]
DEEP = ["dqn", "double_dqn", "reinforce", "ppo"]
REWARDS = ["proposal", "objective", "pbrs", "pbrs_stock"]
LAYOUTS = list(range(10))

_cache = {}


def reference(layout):
    from mdp import StoreMDP
    from evaluation import Reference
    if layout not in _cache:
        _cache[layout] = Reference(StoreMDP(layout))
    return _cache[layout]


def run_one(job):
    algo, reward, layout, episodes, tag, q_init = job
    out_file = RUNS / f"{algo}__{reward}__{layout}{'__' + tag if tag else ''}.json"
    if out_file.exists():
        return str(out_file), 0.0
    import tabular
    from evaluation import simulate
    if algo in DEEP:
        import torch
        torch.set_num_threads(1)
        import deep

    ref = reference(layout)
    mdp = ref.mdp
    curve = []

    def record(ep, greedy_pi, probs=None):
        e = ref.evaluate(greedy_pi)
        row = {"episode": ep, "steps": ep * 200, "score": e["score"],
               "service_level": e["service_level"], "distance": e["distance"]}
        if probs is not None:
            row["score_stochastic"] = ref.evaluate(probs)["score"]
        curve.append(row)

    t0 = time.time()
    extra = {}
    probs = None
    if algo in TABULAR:
        Q, visits = tabular.train(algo, mdp, reward=reward, episodes=episodes, q_init=q_init,
                                  callback=lambda ep, Q: record(ep, Q.argmax(1)))
        greedy = Q.argmax(1)
        # value bias against the optimal values of the reward it trained on
        V_train, _, _, _ = mdp.value_iteration(mdp.reward(reward))
        extra["value_bias"] = float(ref.star["occupancy"] @ (Q.max(1) - V_train))
        extra["coverage"] = float(np.mean(visits > 0))
    elif algo in ("dqn", "double_dqn"):
        net = deep.train_dqn(mdp, reward=reward, double=(algo == "double_dqn"),
                             episodes=episodes, q_init=q_init, callback=record)
        with torch.no_grad():
            greedy = net(deep.all_state_features(mdp)).argmax(1).numpy()
    else:
        trainer = deep.train_ppo if algo == "ppo" else deep.train_reinforce
        net = trainer(mdp, reward=reward, episodes=episodes, callback=record)
        greedy, probs = net.policies(deep.all_state_features(mdp))
    train_time = time.time() - t0

    final = ref.evaluate(greedy)
    if probs is not None:
        final_stoch = ref.evaluate(probs)
        extra["stochastic"] = {k: final_stoch[k] for k in
                               ("score", "objective", "service_level", "distance",
                                "agree_opt_states")}

    sim = simulate(lambda env: int(greedy[mdp.index[env.state_key()]]), layout)
    record_ = {
        "algo": algo, "reward": reward, "layout": layout, "episodes": episodes, "tag": tag,
        "q_init": q_init,
        "train_seconds": train_time,
        "final": final,
        "curve": curve,
        "simulation": {
            "objective_mean": float(sim["objective"].mean()),
            "objective_sem": float(sim["objective"].std(ddof=1) / np.sqrt(len(sim["objective"]))),
            "service_level_mean": float(sim["service_level"].mean()),
            "distance_mean": float(sim["distance"].mean()),
            "action_counts": sim["action_counts"].tolist(),
        },
        "policy": greedy.tolist(),
        **extra,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record_))
    return str(out_file), train_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=TABULAR + DEEP)
    ap.add_argument("--rewards", nargs="+", default=REWARDS)
    ap.add_argument("--layouts", nargs="+", type=int, default=LAYOUTS)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--tag", default="")
    ap.add_argument("--q-init", type=float, default=0.0, help="initial Q-value (tabular, DQN)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    jobs = [(a, r, l, args.episodes, args.tag, args.q_init)
            for a in args.algos for r in args.rewards for l in args.layouts]
    # longest jobs first so the pool does not end on one slow straggler
    jobs.sort(key=lambda j: (j[0] in TABULAR, j[0] != "ppo"))
    print(f"{len(jobs)} runs on {args.workers} workers")
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for i, (path, secs) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"[{i:3d}/{len(jobs)}] {Path(path).name}  {secs:6.1f}s  "
                  f"(elapsed {time.time() - t0:6.0f}s)", flush=True)


if __name__ == "__main__":
    main()
