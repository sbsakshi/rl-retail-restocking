"""
Tabular control: Q-Learning, SARSA, Monte Carlo control, n-step Sarsa and
SARSA(lambda), sharing one training loop.

All five learn a dense Q[768, 7] table over the Markov state
(x, y, carried, stock), the same states the DP model uses, so a learned table
can be compared entry by entry with Q*.

Shared settings (identical for every method so that differences come from
the update rule alone):
    alpha 0.15 (constant), gamma 0.95, Q initialised to q_init = 0
    epsilon-greedy, epsilon linear 1.0 -> 0.05 over the first 75% of
    episodes, then held at 0.05
    600 episodes x 200 steps = 120,000 environment steps

Time limit. The 200-step cut is a property of the experiment, not of the
store, so no method treats it as terminal: every target that reaches the
limit bootstraps from the value of the state the robot is in when it is cut
(Pardo et al., 2018). For Monte Carlo this adds gamma^(T-t) Q(s_T, a_T) to the
return, which is negligible except for the last few dozen steps of an episode.
"""

import numpy as np

from retail_env import RetailRestockEnv

N_ACTIONS = 7
DEFAULTS = dict(episodes=600, max_steps=200, alpha=0.15, gamma=0.95,
                eps_start=1.0, eps_end=0.05, eps_decay_frac=0.75,
                n_step=4, lam=0.8, trace_min=1e-3, eval_every=20, q_init=0.0)


def epsilon_at(ep, cfg):
    decay_eps = max(1, int(cfg["eps_decay_frac"] * cfg["episodes"]))
    frac = min(1.0, ep / decay_eps)
    return cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])


def eps_greedy(q_row, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(N_ACTIONS))
    return int(np.argmax(q_row))


def nstep_target(rewards, start, n, gamma, bootstrap):
    """sum_{j<n} gamma^j r[start+j] + gamma^n * bootstrap (rewards past the end are absent)."""
    G, k = 0.0, 0
    for k, r in enumerate(rewards[start:start + n]):
        G += gamma ** k * r
    return G + gamma ** (k + 1) * bootstrap


def train(algo, mdp, reward="pbrs", seed=None, callback=None, **overrides):
    """Train one tabular agent on the store `mdp` describes.

    algo: "qlearning" | "sarsa" | "mc" | "nstep_sarsa" | "sarsa_lambda"
    callback(ep, Q) is called every cfg["eval_every"] episodes and at the end.
    Returns (Q, visits).
    """
    cfg = {**DEFAULTS, **overrides}
    seed = mdp.layout_seed if seed is None else seed
    rng = np.random.default_rng(seed)
    env = RetailRestockEnv(seed=mdp.layout_seed, reward=reward, gamma=cfg["gamma"],
                           max_steps=cfg["max_steps"])
    # q_init: a constant, or one value per state (used to check that
    # potential-based shaping equals initialising Q with the potential)
    q0 = np.asarray(cfg["q_init"], dtype=float)
    Q = np.zeros((mdp.nS, N_ACTIONS)) + (q0[:, None] if q0.ndim == 1 else q0)
    visits = np.zeros(mdp.nS, dtype=np.int64)
    alpha, gamma, n = cfg["alpha"], cfg["gamma"], cfg["n_step"]
    idx = mdp.index

    for ep in range(cfg["episodes"]):
        if callback and ep % cfg["eval_every"] == 0:
            callback(ep, Q)
        env.reset(seed=seed * 1000 + ep)
        eps = epsilon_at(ep, cfg)
        s = idx[env.state_key()]
        a = eps_greedy(Q[s], eps, rng)
        states, actions, rewards = [s], [a], []
        trace = {}

        for t in range(cfg["max_steps"]):
            visits[s] += 1
            _, r, _, truncated, _ = env.step(a)
            s2 = idx[env.state_key()]
            if algo == "qlearning":
                # off-policy: the target ignores the next action, which is
                # chosen after the update (Sutton & Barto, Fig. 6.6)
                Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
                a2 = eps_greedy(Q[s2], eps, rng)
            else:
                a2 = eps_greedy(Q[s2], eps, rng)    # on-policy targets need A'

            if algo == "sarsa":
                Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            elif algo == "sarsa_lambda":
                delta = r + gamma * Q[s2, a2] - Q[s, a]
                trace[(s, a)] = trace.get((s, a), 0.0) + 1.0      # accumulating trace
                decay = gamma * cfg["lam"]
                kept = {}
                for (ts, ta), e in trace.items():
                    Q[ts, ta] += alpha * delta * e
                    if e * decay >= cfg["trace_min"]:
                        kept[(ts, ta)] = e * decay
                trace = kept
            elif algo in ("mc", "nstep_sarsa"):
                rewards.append(r)
                states.append(s2)
                actions.append(a2)
                if algo == "nstep_sarsa":
                    tau = t - n + 1                                # step whose n-step return is now complete
                    if tau >= 0:
                        G = nstep_target(rewards, tau, n, gamma, Q[s2, a2])
                        st, at = states[tau], actions[tau]
                        Q[st, at] += alpha * (G - Q[st, at])
            s, a = s2, a2
            if truncated:
                break

        if algo == "nstep_sarsa":
            # finish the last n-1 steps, each bootstrapping from the cut state
            T = len(rewards)
            for tau in range(max(0, T - n + 1), T):
                G = nstep_target(rewards, tau, n, gamma, Q[states[T], actions[T]])
                st, at = states[tau], actions[tau]
                Q[st, at] += alpha * (G - Q[st, at])
        elif algo == "mc":
            T = len(rewards)
            G = Q[states[T], actions[T]]                           # bootstrap only at the time limit
            for t in reversed(range(T)):
                G = rewards[t] + gamma * G
                st, at = states[t], actions[t]
                Q[st, at] += alpha * (G - Q[st, at])

    if callback:
        callback(cfg["episodes"], Q)
    env.close()
    return Q, visits
