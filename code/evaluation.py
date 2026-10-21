"""
How every policy in the study is scored.

Exact evaluation. All learners act on the Markov state (x, y, carried,
stock), so any trained agent, Q-table or neural network, can be written as a
policy over the 768 model states and evaluated exactly with
StoreMDP.episode_metrics: expected values over a 200-step episode from
reset, no sampling noise. The model is only a measuring instrument here;
no learner ever sees it.

Normalised score. Stores differ a lot in difficulty (the best achievable
service level ranges from about 0.68 to 0.97 across layouts), so raw
returns from different layouts cannot be averaged. Each policy is placed
between a uniform-random policy (0) and the best possible 200-step
behaviour from reset (1), computed by backward induction:

    score = (J_pi - J_random) / (J_optimal_T - J_random)

where J is the expected 200-step sum of the objective reward.

Simulation. As a cross-check on the model, final policies are also rolled
out in the real environment on 100 demand seeds never used in training.
"""

import numpy as np

from retail_env import RetailRestockEnv, ACTION_NAMES

EVAL_SEED_OFFSET = 90_000      # training uses seed*1000 + episode, episode < 1000
TIE_TOL = 1e-6


class Reference:
    """Planning results for one store layout, computed once and reused."""

    def __init__(self, mdp, gamma=0.95, T=200):
        self.mdp, self.T = mdp, T
        R = mdp.objective()
        self.V_star, self.Q_star, self.pi_star, self.vi_sweeps = mdp.value_iteration(R, gamma)
        self.J_opt_T = mdp.finite_horizon_optimum(R, T)
        self.random = mdp.episode_metrics(np.full((mdp.nS, mdp.nA), 1.0 / mdp.nA), T)
        self.star = mdp.episode_metrics(self.pi_star, T)
        self.optimal_actions = self.Q_star >= self.Q_star.max(axis=1, keepdims=True) - TIE_TOL

    def score(self, J):
        return (J - self.random["objective"]) / (self.J_opt_T - self.random["objective"])

    def evaluate(self, pi):
        """Exact metrics + normalised score + agreement with the optimal policy."""
        m = self.mdp.episode_metrics(pi, self.T)
        probs = self.mdp._as_probs(pi)
        p_opt = np.sum(probs * self.optimal_actions, axis=1)   # prob. of an optimal action
        out = {k: v for k, v in m.items() if k != "occupancy"}
        out["score"] = self.score(m["objective"])
        # where the optimal policy spends its time / where this policy does
        out["agree_opt_states"] = float(self.star["occupancy"] @ p_opt)
        out["agree_own_states"] = float(m["occupancy"] @ p_opt)
        return out


def simulate(act, layout_seed, n_episodes=100, max_steps=200, reward="objective"):
    """Roll a frozen policy out on held-out demand seeds.

    act(env) -> action; it may read env.state_key() or the observation.
    """
    env = RetailRestockEnv(seed=layout_seed, reward=reward, max_steps=max_steps)
    rows, counts = [], np.zeros(len(ACTION_NAMES), dtype=np.int64)
    for i in range(n_episodes):
        env.reset(seed=layout_seed * 1000 + EVAL_SEED_OFFSET + i)
        for _ in range(max_steps):
            a = act(env)
            counts[a] += 1
            env.step(a)
        st = env.stats
        rows.append({
            "objective": st["objective"],
            "service_level": 1 - st["empty_shelf_steps"] / (max_steps * env.n_shelves),
            "distance": st["distance"],
            "restocks": st["restocks"],
            "empty_shelf_steps": st["empty_shelf_steps"],
            "stockouts": st["stockouts"],
        })
    env.close()
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["action_counts"] = counts
    return out
