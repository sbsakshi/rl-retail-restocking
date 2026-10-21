"""
Function-approximation agents: DQN, Double DQN, REINFORCE with a learned
baseline, and PPO.

Common to all four
------------------
Input      the Markov state (x, y, carried, stock_1, stock_2), each scaled to
           [0, 1]. This is the same information the tabular agents and the DP
           model use; it lets the trained network be read out as a policy over
           the 768 model states and evaluated exactly (evaluation.py).
Networks   two hidden layers of 64 ReLU units.
Budget     600 episodes x 200 steps = 120,000 environment steps, the same as
           the tabular agents.
Rewards    multiplied by 0.1 inside the update only (targets of order 10
           rather than 100); everything reported uses the unscaled objective.
Time limit never treated as terminal: targets and returns that reach step 200
           bootstrap from the value of the state the episode was cut in.

Hyperparameters are the usual published defaults for each method (DQN as in
Mnih et al. 2015 scaled to this budget; PPO as in Schulman et al. 2017 and
the Stable-Baselines3 defaults), not tuned per method, so that no algorithm
benefits from a larger tuning effort than the others.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from retail_env import RetailRestockEnv
from tabular import epsilon_at

N_ACTIONS = 7

DQN_DEFAULTS = dict(episodes=600, max_steps=200, gamma=0.95, lr=5e-4, batch_size=64,
                    buffer_size=50_000, learning_starts=1_000, target_every=500,
                    eps_start=1.0, eps_end=0.05, eps_decay_frac=0.75,
                    reward_scale=0.1, grad_clip=10.0, eval_every=20, q_init=0.0)

# REINFORCE has no widely used default. Pooling 10 episodes per update left
# only 60 gradient steps in the budget and the policy stayed at chance, so it
# takes the textbook one step per episode; the policy learning rate was chosen
# from {1e-3, 3e-3, 1e-2} on layouts 0, 4 and 7 (the only setting chosen by trial).
REINFORCE_DEFAULTS = dict(episodes=600, max_steps=200, gamma=0.95, policy_lr=3e-3,
                          value_lr=1e-3, episodes_per_update=1, entropy_coef=0.01,
                          reward_scale=0.1, grad_clip=0.5, eval_every=20)

PPO_DEFAULTS = dict(episodes=600, max_steps=200, gamma=0.95, gae_lambda=0.95, lr=3e-4,
                    rollout_steps=2048, epochs=10, minibatch_size=64, clip_eps=0.2,
                    entropy_coef=0.01, reward_scale=0.1, grad_clip=0.5, eval_every=20)


def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


def features(state, mdp):
    """(x, y, carried, stock...) -> scaled float vector."""
    x, y, c, stock = state
    g = mdp.grid_size - 1
    return [x / g, y / g, c / mdp.carry_capacity] + [s / mdp.stock_max for s in stock]


def all_state_features(mdp):
    return torch.tensor([features(s, mdp) for s in mdp.states], dtype=torch.float32)


def _setup(seed, mdp, reward, cfg):
    torch.manual_seed(seed)
    random.seed(seed)
    env = RetailRestockEnv(seed=mdp.layout_seed, reward=reward, gamma=cfg["gamma"],
                           max_steps=cfg["max_steps"])
    return env, np.random.default_rng(seed), all_state_features(mdp)


def _obs(env, mdp):
    return np.array(features(env.state_key(), mdp), dtype=np.float32)


# ---------------------------------------------------------------------- #
#  DQN / Double DQN
# ---------------------------------------------------------------------- #
def train_dqn(mdp, reward="pbrs", double=False, seed=None, callback=None, **overrides):
    """callback(ep, greedy_policy) every eval_every episodes. Returns the online net."""
    cfg = {**DQN_DEFAULTS, **overrides}
    seed = mdp.layout_seed if seed is None else seed
    env, rng, X_all = _setup(seed, mdp, reward, cfg)
    n_in = X_all.shape[1]
    online, target = mlp(n_in, N_ACTIONS), mlp(n_in, N_ACTIONS)
    if cfg["q_init"]:
        # start the network's outputs near q_init (same role as Q_0 for a
        # table); the default leaves PyTorch's initialisation, outputs ~ 0
        with torch.no_grad():
            online[-1].bias.fill_(cfg["q_init"] * cfg["reward_scale"])
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=cfg["lr"])

    cap = cfg["buffer_size"]
    S = np.zeros((cap, n_in), np.float32); S2 = np.zeros((cap, n_in), np.float32)
    A = np.zeros(cap, np.int64); Rw = np.zeros(cap, np.float32)
    size = ptr = steps = 0

    def greedy():
        with torch.no_grad():
            return online(X_all).argmax(1).numpy()

    for ep in range(cfg["episodes"]):
        if callback and ep % cfg["eval_every"] == 0:
            callback(ep, greedy())
        env.reset(seed=seed * 1000 + ep)
        eps = epsilon_at(ep, cfg)
        s = _obs(env, mdp)
        for t in range(cfg["max_steps"]):
            if rng.random() < eps:
                a = int(rng.integers(N_ACTIONS))
            else:
                with torch.no_grad():
                    a = int(online(torch.from_numpy(s)).argmax())
            _, r, _, truncated, _ = env.step(a)
            s2 = _obs(env, mdp)
            # no done flag: the environment never terminates and the time
            # limit is not a terminal state, so every target bootstraps
            S[ptr], A[ptr], Rw[ptr], S2[ptr] = s, a, r * cfg["reward_scale"], s2
            ptr = (ptr + 1) % cap
            size = min(size + 1, cap)
            steps += 1
            s = s2

            if steps >= cfg["learning_starts"]:
                b = rng.integers(size, size=cfg["batch_size"])
                sb, ab = torch.from_numpy(S[b]), torch.from_numpy(A[b])
                rb, s2b = torch.from_numpy(Rw[b]), torch.from_numpy(S2[b])
                with torch.no_grad():
                    if double:   # online net selects, target net evaluates
                        a_next = online(s2b).argmax(1, keepdim=True)
                        q_next = target(s2b).gather(1, a_next).squeeze(1)
                    else:
                        q_next = target(s2b).max(1).values
                    y = rb + cfg["gamma"] * q_next
                q = online(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), cfg["grad_clip"])
                opt.step()
            if steps % cfg["target_every"] == 0:
                target.load_state_dict(online.state_dict())
            if truncated:
                break

    if callback:
        callback(cfg["episodes"], greedy())
    env.close()
    return online


# ---------------------------------------------------------------------- #
#  Policy-gradient helpers
# ---------------------------------------------------------------------- #
class ActorCritic:
    def __init__(self, n_in):
        self.pi, self.v = mlp(n_in, N_ACTIONS), mlp(n_in, 1)

    def dist(self, x):
        return Categorical(logits=self.pi(x))

    def value(self, x):
        return self.v(x).squeeze(-1)

    def policies(self, X_all):
        """(greedy policy, action-probability matrix) over all model states."""
        with torch.no_grad():
            logits = self.pi(X_all)
        return logits.argmax(1).numpy(), torch.softmax(logits, 1).numpy()


def _collect_episode(env, mdp, net, seed):
    env.reset(seed=seed)
    obs, acts, rews = [], [], []
    s = _obs(env, mdp)
    for _ in range(env.max_steps):
        with torch.no_grad():
            a = int(net.dist(torch.from_numpy(s)).sample())
        _, r, _, truncated, _ = env.step(a)
        obs.append(s); acts.append(a); rews.append(r)
        s = _obs(env, mdp)
        if truncated:
            break
    return np.array(obs), np.array(acts), np.array(rews, np.float32), s


# ---------------------------------------------------------------------- #
#  REINFORCE with a learned baseline
# ---------------------------------------------------------------------- #
def train_reinforce(mdp, reward="pbrs", seed=None, callback=None, **overrides):
    """callback(ep, greedy_policy, probs). Returns the ActorCritic."""
    cfg = {**REINFORCE_DEFAULTS, **overrides}
    seed = mdp.layout_seed if seed is None else seed
    env, rng, X_all = _setup(seed, mdp, reward, cfg)
    net = ActorCritic(X_all.shape[1])
    opt_pi = torch.optim.Adam(net.pi.parameters(), lr=cfg["policy_lr"])
    opt_v = torch.optim.Adam(net.v.parameters(), lr=cfg["value_lr"])
    g = cfg["gamma"]
    batch_obs, batch_act, batch_ret = [], [], []

    for ep in range(cfg["episodes"]):
        if callback and ep % cfg["eval_every"] == 0:
            callback(ep, *net.policies(X_all))
        obs, acts, rews, s_last = _collect_episode(env, mdp, net, seed * 1000 + ep)
        rews = rews * cfg["reward_scale"]
        with torch.no_grad():
            G = float(net.value(torch.from_numpy(s_last)))     # tail beyond the time limit
        returns = np.zeros_like(rews)
        for t in reversed(range(len(rews))):
            G = rews[t] + g * G
            returns[t] = G
        batch_obs.append(obs); batch_act.append(acts); batch_ret.append(returns)

        if (ep + 1) % cfg["episodes_per_update"] == 0:
            x = torch.from_numpy(np.concatenate(batch_obs))
            a = torch.from_numpy(np.concatenate(batch_act))
            ret = torch.from_numpy(np.concatenate(batch_ret))
            v = net.value(x)
            adv = ret - v.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            dist = net.dist(x)
            loss_pi = -(dist.log_prob(a) * adv).mean() - cfg["entropy_coef"] * dist.entropy().mean()
            opt_pi.zero_grad(); loss_pi.backward()
            nn.utils.clip_grad_norm_(net.pi.parameters(), cfg["grad_clip"]); opt_pi.step()
            loss_v = nn.functional.mse_loss(v, ret)
            opt_v.zero_grad(); loss_v.backward()
            nn.utils.clip_grad_norm_(net.v.parameters(), cfg["grad_clip"]); opt_v.step()
            batch_obs, batch_act, batch_ret = [], [], []

    if callback:
        callback(cfg["episodes"], *net.policies(X_all))
    env.close()
    return net


# ---------------------------------------------------------------------- #
#  PPO (clipped surrogate, GAE)
# ---------------------------------------------------------------------- #
def gae(rewards, values, last_value, gamma, lam):
    """Advantages for one episode segment that was cut by the time limit.

    values[t] = V(s_t); last_value = V(s_T) for the state the segment ended in.
    """
    adv = np.zeros_like(rewards)
    running = 0.0
    nxt = last_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * nxt - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
        nxt = values[t]
    return adv


def train_ppo(mdp, reward="pbrs", seed=None, callback=None, **overrides):
    """callback(ep, greedy_policy, probs). Returns the ActorCritic."""
    cfg = {**PPO_DEFAULTS, **overrides}
    seed = mdp.layout_seed if seed is None else seed
    env, rng, X_all = _setup(seed, mdp, reward, cfg)
    net = ActorCritic(X_all.shape[1])
    params = list(net.pi.parameters()) + list(net.v.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    g, lam = cfg["gamma"], cfg["gae_lambda"]
    buf = {"obs": [], "act": [], "logp": [], "adv": [], "ret": []}
    steps_in_buf = 0

    for ep in range(cfg["episodes"]):
        if callback and ep % cfg["eval_every"] == 0:
            callback(ep, *net.policies(X_all))
        obs, acts, rews, s_last = _collect_episode(env, mdp, net, seed * 1000 + ep)
        rews = rews * cfg["reward_scale"]
        with torch.no_grad():
            x = torch.from_numpy(obs)
            values = net.value(x).numpy()
            logp = net.dist(x).log_prob(torch.from_numpy(acts)).numpy()
            last_v = float(net.value(torch.from_numpy(s_last)))
        adv = gae(rews, values, last_v, g, lam)
        for k, v in (("obs", obs), ("act", acts), ("logp", logp),
                     ("adv", adv), ("ret", adv + values)):
            buf[k].append(v)
        steps_in_buf += len(rews)

        if steps_in_buf >= cfg["rollout_steps"] or ep == cfg["episodes"] - 1:
            data = {k: torch.from_numpy(np.concatenate(v)) for k, v in buf.items()}
            N = len(data["act"])
            for _ in range(cfg["epochs"]):
                perm = rng.permutation(N)
                for start in range(0, N, cfg["minibatch_size"]):
                    mb = torch.from_numpy(perm[start:start + cfg["minibatch_size"]])
                    a_mb = data["adv"][mb]
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                    dist = net.dist(data["obs"][mb])
                    ratio = torch.exp(dist.log_prob(data["act"][mb]) - data["logp"][mb])
                    surr = torch.min(ratio * a_mb,
                                     torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * a_mb)
                    loss = (-surr.mean()
                            + 0.5 * nn.functional.mse_loss(net.value(data["obs"][mb]), data["ret"][mb])
                            - cfg["entropy_coef"] * dist.entropy().mean())
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
                    opt.step()
            buf = {k: [] for k in buf}
            steps_in_buf = 0

    if callback:
        callback(cfg["episodes"], *net.policies(X_all))
    env.close()
    return net
