"""
Exact finite model of RetailRestockEnv, and the planning tools built on it.

State s = (x, y, carried, stock_1..stock_n). For the default store this is
4*4 * 3 * 4**2 = 768 states. Nothing else in the observation (time since
visit, time of day) enters the transition or the objective, so this is a
Markov state and planning on it is exact, not an approximation.

Transitions are deterministic given the action except for demand: each
shelf with stock left loses one unit with probability p, independently. So
every (s, a) has at most 2**n_shelves successors, stored densely:

    NS[s, a, b]   successor index of branch b
    P[s, a, b]    its probability (0 for padding)
    C[c][s, a, b] value of reward component c on that branch

Any reward, including potential-based shaping, is a linear combination of
the components, so one model serves every reward variant in the study.

Policies are either an int array (deterministic) or an (nS, nA) array of
action probabilities. `episode_metrics` propagates the exact state
distribution through a T-step episode, giving the expected value of every
metric with no sampling noise.
"""

import itertools
import numpy as np

from retail_env import (COMPONENTS, OBJECTIVE_WEIGHTS, REWARD_PRESETS,
                        shelf_layout, potential_terms)


class StoreMDP:
    def __init__(self, layout_seed, grid_size=4, n_shelves=2, stock_max=3,
                 carry_capacity=2, demand_prob=0.12, depot=(0, 0)):
        self.layout_seed = layout_seed
        self.grid_size, self.n_shelves = grid_size, n_shelves
        self.stock_max, self.carry_capacity = stock_max, carry_capacity
        self.demand_prob, self.depot = demand_prob, depot
        self.shelves = shelf_layout(grid_size, n_shelves, layout_seed, depot)

        self.states = [(x, y, c, st)
                       for x in range(grid_size) for y in range(grid_size)
                       for c in range(carry_capacity + 1)
                       for st in itertools.product(range(stock_max + 1), repeat=n_shelves)]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.nS, self.nA, self.nB = len(self.states), 7, 2 ** n_shelves
        self.start = self.index[(depot[0], depot[1], carry_capacity,
                                 (stock_max,) * n_shelves)]
        self._build()

    # ------------------------------------------------------------------ #
    def _deterministic(self, s, a):
        """Action effect before demand. Mirrors RetailRestockEnv.step()."""
        x, y, carried, stock = s
        stock = list(stock)
        comp = dict.fromkeys(COMPONENTS, 0)
        comp["step"] = 1
        dx, dy = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}.get(a, (0, 0))
        nx, ny = x + dx, y + dy
        if (dx or dy) and 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            x, y = nx, ny
            comp["moved"] = 1
        elif a == 4:
            done = False
            if (x, y) in self.shelves and carried > 0:
                i = self.shelves.index((x, y))
                units = min(carried, self.stock_max - stock[i])
                if units > 0:
                    comp["restock_proactive" if stock[i] > 0 else "restock_reactive"] = 1
                    stock[i] += units
                    carried -= units
                    done = True
            if not done:
                comp["invalid"] = 1
        elif a == 5:
            if (x, y) == self.depot:
                carried = self.carry_capacity
            else:
                comp["invalid"] = 1
        tx, ty = self.shelves[int(np.argmin(stock))]
        d_prev = abs(s[0] - tx) + abs(s[1] - ty)
        comp["closer"] = int(abs(x - tx) + abs(y - ty) < d_prev)
        return (x, y, carried), stock, comp

    def _build(self):
        nS, nA, nB, p = self.nS, self.nA, self.nB, self.demand_prob
        self.NS = np.zeros((nS, nA, nB), dtype=np.int64)
        self.P = np.zeros((nS, nA, nB))
        self.C = {c: np.zeros((nS, nA, nB)) for c in COMPONENTS}
        self.units = np.zeros((nS, nA))          # units moved onto a shelf
        self.onsets = np.zeros((nS, nA, nB))     # shelves that run out on this branch
        terms = [potential_terms(s[:2], s[3], self.shelves) for s in self.states]
        self.phi = {k: np.array([t[k] for t in terms]) for k in terms[0]}

        for si, s in enumerate(self.states):
            for a in range(nA):
                (x, y, c), stock, comp = self._deterministic(s, a)
                self.units[si, a] = s[2] - c if a == 4 else 0
                for b, drops in enumerate(itertools.product((0, 1), repeat=self.n_shelves)):
                    prob, new = 1.0, list(stock)
                    for i, d in enumerate(drops):
                        if new[i] == 0:
                            if d:            # an empty shelf cannot lose stock:
                                prob = 0.0   # fold this branch into d=0
                            continue
                        prob *= p if d else 1 - p
                        new[i] -= d
                    if prob == 0.0:
                        continue
                    self.NS[si, a, b] = self.index[(x, y, c, tuple(new))]
                    self.P[si, a, b] = prob
                    for k in COMPONENTS:
                        self.C[k][si, a, b] = comp[k]
                    self.C["n_empty"][si, a, b] = sum(v == 0 for v in new)
                    self.onsets[si, a, b] = sum(o > 0 and v == 0 for o, v in zip(stock, new))
        assert np.allclose(self.P.sum(axis=2), 1.0)

    # ------------------------------------------------------------------ #
    def reward(self, spec, gamma=0.95):
        """R[s, a, b] for a preset name or a {"weights", "potential"} dict."""
        spec = REWARD_PRESETS[spec] if isinstance(spec, str) else spec
        R = sum(w * self.C[c] for c, w in spec["weights"].items())
        for k, w in spec.get("potential", {}).items():
            phi = w * self.phi[k]
            R = R + gamma * phi[self.NS] - phi[:, None, None]
        return R

    def objective(self):
        return self.reward({"weights": OBJECTIVE_WEIGHTS})

    def q_from_v(self, V, R, gamma):
        return np.sum(self.P * (R + gamma * V[self.NS]), axis=2)

    def value_iteration(self, R, gamma=0.95, tol=1e-10, max_sweeps=100000):
        V = np.zeros(self.nS)
        for sweep in range(1, max_sweeps + 1):
            Q = self.q_from_v(V, R, gamma)
            V_new = Q.max(axis=1)
            delta = np.max(np.abs(V_new - V))
            V = V_new
            if delta < tol:
                break
        Q = self.q_from_v(V, R, gamma)
        return V, Q, Q.argmax(axis=1), sweep

    def policy_matrix(self, pi):
        """(nS, nS) state-transition matrix under policy pi."""
        pi = self._as_probs(pi)
        M = np.zeros((self.nS, self.nS))
        rows = np.repeat(np.arange(self.nS), self.nB)
        for a in range(self.nA):
            w = (pi[:, a:a + 1] * self.P[:, a, :]).ravel()
            np.add.at(M, (rows, self.NS[:, a, :].ravel()), w)
        return M

    def expected_reward(self, pi, R):
        """r_pi[s] = E[R | s, a ~ pi(s)]."""
        return np.sum(self._as_probs(pi) * np.sum(self.P * R, axis=2), axis=1)

    def evaluate(self, pi, R, gamma=0.95):
        """Exact V^pi by solving (I - gamma P_pi) V = r_pi."""
        M = self.policy_matrix(pi)
        return np.linalg.solve(np.eye(self.nS) - gamma * M, self.expected_reward(pi, R))

    def policy_iteration(self, R, gamma=0.95, max_iters=1000):
        pi = np.zeros(self.nS, dtype=np.int64)
        for it in range(1, max_iters + 1):
            V = self.evaluate(pi, R, gamma)
            Q = self.q_from_v(V, R, gamma)
            # keep the current action unless another is strictly better,
            # otherwise ties between equal actions make PI cycle
            best = Q.max(axis=1)
            keep = Q[np.arange(self.nS), pi] >= best - 1e-9
            new_pi = np.where(keep, pi, Q.argmax(axis=1))
            if np.array_equal(new_pi, pi):
                return V, Q, pi, it
            pi = new_pi
        return V, Q, pi, it

    def finite_horizon_optimum(self, R, T):
        """Best achievable expected T-step (undiscounted) return from the start state."""
        V = np.zeros(self.nS)
        for _ in range(T):
            V = np.max(np.sum(self.P * (R + V[self.NS]), axis=2), axis=1)
        return V[self.start]

    # ------------------------------------------------------------------ #
    def _as_probs(self, pi):
        pi = np.asarray(pi)
        if pi.ndim == 1:
            out = np.zeros((self.nS, self.nA))
            out[np.arange(self.nS), pi] = 1.0
            return out
        return pi

    def episode_metrics(self, pi, T=200):
        """Exact expected per-episode metrics for a T-step episode from reset."""
        probs = self._as_probs(pi)
        M = self.policy_matrix(probs)
        PR = lambda X: np.sum(probs * np.sum(self.P * X, axis=2), axis=1)
        per_state = {
            "objective": PR(self.objective()),
            "empty": PR(self.C["n_empty"]),
            "distance": PR(self.C["moved"]),
            "restocks": PR(self.C["restock_proactive"] + self.C["restock_reactive"]),
            "proactive": PR(self.C["restock_proactive"]),
            "units": np.sum(probs * self.units, axis=1),
            "onsets": PR(self.onsets),
        }
        d = np.zeros(self.nS)
        d[self.start] = 1.0
        occupancy = np.zeros(self.nS)
        for _ in range(T):
            occupancy += d
            d = d @ M
        tot = {k: float(occupancy @ v) for k, v in per_state.items()}
        return {
            "objective": tot["objective"],
            "service_level": 1.0 - tot["empty"] / (T * self.n_shelves),
            "distance": tot["distance"],
            "restocks": tot["restocks"],
            "proactive_share": tot["proactive"] / max(tot["restocks"], 1e-12),
            "load_factor": tot["units"] / max(tot["restocks"], 1e-12) / self.carry_capacity,
            # mean length of a stockout: empty shelf-steps per shelf that ran out
            # (a shelf still empty at step T counts as a truncated spell)
            "stockout_duration": tot["empty"] / max(tot["onsets"], 1e-12),
            "occupancy": occupancy / T,
        }

    def greedy_from_q_table(self, Q_table):
        """Q_table: (nS, nA) array indexed like self.states -> greedy policy."""
        return np.asarray(Q_table).argmax(axis=1)
