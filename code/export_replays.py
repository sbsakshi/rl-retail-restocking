"""
Replays for the 3D viewer (viewer/index.html).

    python export_replays.py        # a few seconds; writes results/replays/

Nothing is retrained. Every run file stores its greedy policy for all 768
states, so each replay is that policy driven through the real simulator.
For every store layout, every agent faces the same held-out demand sequence
(seed layout*1000 + 90000, never used in training), so differences between
replays come from the policies alone.

Agents shown: the exact optimum, the nine learners trained on the Proposal
reward (the reward the report's learning curves use), and Q-Learning and
SARSA(lambda) trained on the Objective reward with Q0 = -150, the initial
value study of Section VI-D.

Per step the replay records the *objective* reward (-8 per empty shelf,
-0.1 per move), the quantity every agent is scored on, whatever it trained on.

Also writes details.json: per-agent description, configuration, exact
results on every layout, and the figures for the viewer's side panel.
"""

import json
from pathlib import Path

import numpy as np

import analyze
import deep
import tabular
from evaluation import Reference, EVAL_SEED_OFFSET
from mdp import StoreMDP
from retail_env import RetailRestockEnv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "replays"
LAYOUTS = range(10)
ACTION_LABELS = ["move up", "move down", "move left", "move right", "restock", "reload", "wait"]

# (key, run selector, label); selector = (algo, reward, tag) or "dp"
AGENTS = [
    ("dp_optimum", "dp", "Exact optimum (DP)"),
    ("dqn", ("dqn", "proposal", ""), "DQN"),
    ("double_dqn", ("double_dqn", "proposal", ""), "Double DQN"),
    ("ppo", ("ppo", "proposal", ""), "PPO"),
    ("reinforce", ("reinforce", "proposal", ""), "REINFORCE + baseline"),
    ("sarsa_lambda", ("sarsa_lambda", "proposal", ""), "SARSA(λ)"),
    ("nstep_sarsa", ("nstep_sarsa", "proposal", ""), "4-step Sarsa"),
    ("sarsa", ("sarsa", "proposal", ""), "SARSA"),
    ("mc", ("mc", "proposal", ""), "Monte Carlo control"),
    ("qlearning", ("qlearning", "proposal", ""), "Q-Learning"),
    ("sarsa_lambda_q150", ("sarsa_lambda", "objective", "init-150"), "SARSA(λ) · Q₀ = −150"),
    ("qlearning_q150", ("qlearning", "objective", "init-150"), "Q-Learning · Q₀ = −150"),
]

FAMILY = {
    "dp_optimum": "Exact planning", "qlearning": "Tabular TD (off-policy)",
    "sarsa": "Tabular TD (on-policy)", "nstep_sarsa": "Tabular multi-step TD",
    "sarsa_lambda": "Tabular TD with eligibility traces", "mc": "Tabular Monte Carlo",
    "dqn": "Deep value-based", "double_dqn": "Deep value-based",
    "reinforce": "Policy gradient", "ppo": "Policy gradient (actor-critic)",
}
SUMMARY = {
    "dp_optimum": "Value iteration on the exact 768-state model of the store, with the objective reward and "
                  "gamma = 0.95. It is given the transition probabilities rather than learning them, so it is the "
                  "reference every learner is measured against (normalised score close to 1).",
    "qlearning": "Moves Q(s,a) toward r + gamma max Q(s',.). Off-policy, one step at a time. With Q0 = 0 it keeps "
                 "most of its initial optimism after 120,000 steps and is the weakest TD learner.",
    "sarsa": "Moves Q(s,a) toward r + gamma Q(s',a') for the action actually taken next. On-policy, one step at a time.",
    "nstep_sarsa": "Uses four observed rewards plus gamma^4 Q(s_t+4, a_t+4) as its target, so each update passes "
                   "credit back four steps.",
    "sarsa_lambda": "Applies each TD error to all recently visited state-action pairs through accumulating "
                    "eligibility traces (lambda = 0.8). The strongest tabular method.",
    "mc": "Moves Q(s,a) toward the observed discounted return of the episode; never bootstraps inside an episode.",
    "dqn": "A neural Q-function trained from a replay buffer against a periodically copied target network. "
           "The strongest learner: within a few percent of the optimum on the Proposal reward.",
    "double_dqn": "DQN where the online network chooses the next action and the target network evaluates it, "
                  "to reduce overestimation. Statistically indistinguishable from DQN here.",
    "reinforce": "Raises the log-probability of each action in proportion to its return minus a learned "
                 "baseline. The replay shows its greedy (argmax) policy, which scores far below the stochastic "
                 "policy it actually learned.",
    "ppo": "Clipped-objective actor-critic with generalised advantage estimation, reusing each 2048-step batch "
           "for 10 epochs. Strong on some layouts, stalls on others.",
}
RULE = {
    "dp_optimum": "V(s) <- max_a sum_s' P(s'|s,a) [ r(s,a,s') + gamma V(s') ]",
    "qlearning": "Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]",
    "sarsa": "Q(s,a) <- Q(s,a) + alpha [ r + gamma Q(s',a') - Q(s,a) ]",
    "nstep_sarsa": "Q(s_t,a_t) <- Q + alpha [ r_t+1 + ... + gamma^3 r_t+4 + gamma^4 Q(s_t+4,a_t+4) - Q ]",
    "sarsa_lambda": "delta = r + gamma Q(s',a') - Q(s,a);  e(s,a) += 1;  Q += alpha delta e;  e *= gamma lambda",
    "mc": "Q(s_t,a_t) <- Q + alpha [ G_t - Q ],  G_t = r_t+1 + gamma G_t+1",
    "dqn": "loss = Huber( r + gamma max_a' Q_target(s',a') - Q(s,a) )",
    "double_dqn": "loss = Huber( r + gamma Q_target(s', argmax_a' Q(s',a')) - Q(s,a) )",
    "reinforce": "grad = E[ (G_t - V(s_t)) grad log pi(a_t|s_t) ] + entropy bonus",
    "ppo": "L = E[ min(rho A, clip(rho, 0.8, 1.2) A) ],  rho = pi(a|s) / pi_old(a|s),  A from GAE(0.95)",
}


def config_for(key, algo, reward, tag):
    if key == "dp_optimum":
        return [["States", "768 (x, y, carried, stock1, stock2)"], ["Reward", "objective"],
                ["Discount", "0.95"], ["Stopping rule", "max change < 1e-10"]]
    if algo in analyze.ALGOS[:5]:
        c = dict(tabular.DEFAULTS)
        rows = [["Learning rate alpha", c["alpha"]], ["Discount", c["gamma"]],
                ["Initial Q-value", -150 if tag == "init-150" else c["q_init"]],
                ["Exploration", "epsilon 1.0 -> 0.05 over 450 episodes"]]
        if algo == "nstep_sarsa":
            rows.append(["n", c["n_step"]])
        if algo == "sarsa_lambda":
            rows.append(["lambda", c["lam"]])
    elif algo in ("dqn", "double_dqn"):
        c = deep.DQN_DEFAULTS
        rows = [["Network", "2 x 64 ReLU"], ["Adam learning rate", c["lr"]], ["Batch", c["batch_size"]],
                ["Replay buffer", c["buffer_size"]], ["Target copy every", f"{c['target_every']} steps"],
                ["Exploration", "epsilon 1.0 -> 0.05 over 450 episodes"]]
    elif algo == "ppo":
        c = deep.PPO_DEFAULTS
        rows = [["Network", "2 x 64 ReLU (actor and critic)"], ["Adam learning rate", c["lr"]],
                ["Rollout", f"{c['rollout_steps']} steps"], ["Epochs", c["epochs"]],
                ["Clip", c["clip_eps"]], ["GAE lambda", c["gae_lambda"]]]
    else:
        c = deep.REINFORCE_DEFAULTS
        rows = [["Network", "2 x 64 ReLU (policy and baseline)"], ["Policy learning rate", c["policy_lr"]],
                ["Baseline learning rate", c["value_lr"]], ["Updates", "one per episode"]]
    return [["Training reward", reward], ["Budget", "600 episodes x 200 steps"]] + rows


def rollout(policy, layout, mdp):
    env = RetailRestockEnv(seed=layout, reward="objective")
    env.reset(seed=layout * 1000 + EVAL_SEED_OFFSET)
    names = [f"S{i + 1}" for i in range(env.n_shelves)]
    snap = lambda: {n: int(v) for n, v in zip(names, env.stock)}
    steps = [{"t": 0, "robot_pos": list(env.robot_pos), "stock_levels": snap(),
              "carried": int(env.carried), "action": None, "reward": 0.0, "restocked_shelf": None}]
    for t in range(1, env.max_steps + 1):
        a = int(policy[mdp.index[env.state_key()]])
        before = env.stats["restocks"]
        pos = tuple(env.robot_pos)
        _, _, _, _, info = env.step(a)
        restocked = names[env.shelf_positions.index(pos)] if env.stats["restocks"] > before else None
        steps.append({"t": t, "robot_pos": list(env.robot_pos), "stock_levels": snap(),
                      "carried": int(env.carried), "action": ACTION_LABELS[a],
                      "reward": round(info["objective"], 4), "restocked_shelf": restocked})
    return env, steps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    runs = {"": analyze.enrich(analyze.load()), "init-150": analyze.enrich(analyze.load(tag="init-150"))}
    index, methods = [], {}
    dp = json.loads((ROOT / "results/dp_audit.json").read_text())

    for layout in LAYOUTS:
        mdp = StoreMDP(layout)
        ref = Reference(mdp)
        for key, sel, label in AGENTS:
            if sel == "dp":
                policy, final = ref.pi_star, ref.evaluate(ref.pi_star)
                algo = reward = tag = None
            else:
                algo, reward, tag = sel
                run = runs[tag][(algo, reward)][layout]
                policy, final = np.array(run["policy"]), dict(run["final"])
            env, steps = rollout(policy, layout, mdp)
            ret = round(sum(s["reward"] for s in steps[1:]), 4)
            fname = f"{key}__L{layout}.json"
            (OUT / fname).write_text(json.dumps({
                "algorithm": key, "label": label, "layout": layout,
                "grid_size": [env.grid_size, env.grid_size], "depot_pos": list(env.depot_pos),
                "shelf_positions": {f"S{i + 1}": list(p) for i, p in enumerate(env.shelf_positions)},
                "stock_max": env.stock_max, "carry_capacity": env.carry_capacity,
                "episode_seed": layout * 1000 + EVAL_SEED_OFFSET,
                "episode_return": ret,
                "eval_mean_return": round(final["objective"], 2),
                "score": round(final["score"], 4),
                "steps": steps,
            }))
            index.append({"file": fname, "algorithm": key, "label": label, "layout": layout,
                          "optimal_service": round(dp[layout]["rewards"]["objective"]["service_level"], 3)})
            m = methods.setdefault(key, {
                "label": label, "family": FAMILY[key.replace("_q150", "")],
                "summary": SUMMARY[key.replace("_q150", "")] + (
                    " This version was trained on the Objective reward starting from Q0 = -150, the best "
                    "initial value found in the report (Section VI-D)." if key.endswith("_q150") else ""),
                "update_rule": RULE[key.replace("_q150", "")],
                "config": config_for(key, algo, reward, tag),
                "layouts": [], "figures": [],
            })
            m["layouts"].append({k: round(float(final[k]), 4) for k in
                                 ("score", "objective", "service_level", "distance",
                                  "stockout_duration", "agree_opt_states") if k in final})

    figs = {
        "tabular": [("results/figures/curves_proposal.png", "Learning curves, Proposal reward (Fig. 3)"),
                    ("results/figures/init.png", "Initial-value study (Fig. 6)"),
                    ("results/figures/diagnostics_objective.png", "Optimism left in the tables (Fig. 5)")],
        "deep": [("results/figures/curves_proposal.png", "Learning curves, Proposal reward (Fig. 3)"),
                 ("results/figures/frontier_proposal.png", "Learned policies on the exact frontier (Fig. 4)")],
        "dp": [("results/figures/layouts.png", "The ten store layouts (Fig. 1)"),
               ("results/figures/frontier_proposal.png", "Exact service/travel frontier (Fig. 4)")],
    }
    for key, m in methods.items():
        group = "dp" if key == "dp_optimum" else ("deep" if key in ("dqn", "double_dqn", "ppo", "reinforce") else "tabular")
        m["figures"] = [{"src": s, "caption": c} for s, c in figs[group] + [
            ("results/figures/reward_effect.png", "Score of every learner under each training reward (Fig. 2)")]]
        for f in m["figures"]:
            assert (ROOT / f["src"]).exists(), f["src"]

    details = {
        "methods": methods,
        "environment": [
            ["Store", "4 x 4 grid, depot at (0, 0), two shelves of capacity 3"],
            ["Robot", "carries 2 units; 7 actions (4 moves, restock, reload, wait)"],
            ["Demand", "each stocked shelf loses a unit with probability 0.12 per step"],
            ["Episode", "200 steps; the cut is a time limit, not a terminal state"],
            ["Objective reward", "-8 per empty shelf per step, -0.1 per move"],
            ["Score", "0 = uniform-random policy, 1 = best achievable 200-step behaviour"],
            ["Metrics", "exact expected values over a 200-step episode (no sampling noise)"],
        ],
    }
    (OUT / "details.json").write_text(json.dumps(details, indent=1, ensure_ascii=False))
    (OUT / "index.json").write_text(json.dumps({"replays": index}, indent=1, ensure_ascii=False))
    print(f"{len(index)} replays for {len(LAYOUTS)} layouts -> {OUT}")


if __name__ == "__main__":
    main()
