"""
Checks that the measuring instruments are right before any result is trusted.

    python -m pytest tests -q        (or: python tests/test_core.py)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retail_env import RetailRestockEnv
from mdp import StoreMDP
from evaluation import Reference, simulate
from tabular import nstep_target
import review2_env


def test_proposal_reward_reproduces_review2_environment():
    for layout in range(3):
        old = review2_env.RetailRestockEnv(seed=layout)
        new = RetailRestockEnv(seed=layout, reward="proposal")
        assert old.shelf_positions == new.shelf_positions
        rng = np.random.default_rng(layout)
        for ep in range(5):
            old.reset(seed=ep); new.reset(seed=ep)
            for _ in range(200):
                a = int(rng.integers(7))
                o1, r1, *_ = old.step(a)
                o2, r2, *_ = new.step(a)
                assert np.allclose(o1, o2) and abs(r1 - r2) < 1e-9


def _force(env, s):
    env.robot_pos = [s[0], s[1]]
    env.carried = s[2]
    env.stock = np.array(s[3], dtype=np.int32)


def test_model_matches_simulator():
    """Every sampled successor exists in the model with the same reward, for every reward."""
    m = StoreMDP(3)
    rng = np.random.default_rng(0)
    for reward in ("proposal", "objective", "pbrs", "pbrs_stock"):
        env, R = RetailRestockEnv(seed=3, reward=reward), m.reward(reward)
        for k in range(1500):
            s, a = m.states[rng.integers(m.nS)], int(rng.integers(7))
            env.reset(seed=k); _force(env, s)
            _, r, *_ = env.step(a)
            si, s2 = m.index[s], m.index[env.state_key()]
            branch = [b for b in range(m.nB) if m.P[si, a, b] > 0 and m.NS[si, a, b] == s2]
            assert branch, (s, a)
            assert abs(r - R[si, a, branch[0]]) < 1e-9


def test_model_transition_frequencies():
    """Empirical successor frequencies agree with P(s'|s,a) (chi-square-style tolerance)."""
    m = StoreMDP(1)
    env = RetailRestockEnv(seed=1)
    s = (1, 3, 2, (2, 1))                  # robot on a shelf, both shelves partly stocked
    si, a, n = m.index[s], 4, 20000
    counts = {}
    for k in range(n):
        env.reset(seed=k); _force(env, s); env.step(a)
        j = m.index[env.state_key()]
        counts[j] = counts.get(j, 0) + 1
    for b in range(m.nB):
        p = m.P[si, a, b]
        if p > 0:
            f = counts.get(m.NS[si, a, b], 0) / n
            assert abs(f - p) < 4 * np.sqrt(p * (1 - p) / n), (b, f, p)


def test_potential_shaping_preserves_optimal_policy():
    for layout in range(3):
        m = StoreMDP(layout)
        _, Q, _, _ = m.value_iteration(m.objective())
        optimal = Q >= Q.max(1, keepdims=True) - 1e-6
        for reward in ("pbrs", "pbrs_stock"):
            _, _, pi, _ = m.value_iteration(m.reward(reward))
            assert optimal[np.arange(m.nS), pi].all()


def test_value_and_policy_iteration_agree_up_to_ties():
    m = StoreMDP(0)
    R = m.objective()
    V1, Q1, _, _ = m.value_iteration(R)
    V2, _, pi2, _ = m.policy_iteration(R)
    assert np.abs(V1 - V2).max() < 1e-6
    optimal = Q1 >= Q1.max(1, keepdims=True) - 1e-6
    assert optimal[np.arange(m.nS), pi2].all()


def test_exact_metrics_match_simulation():
    ref = Reference(StoreMDP(2))
    pi = ref.pi_star
    exact = ref.mdp.episode_metrics(pi)
    sim = simulate(lambda env: int(pi[ref.mdp.index[env.state_key()]]), 2, n_episodes=300)
    sem = sim["objective"].std(ddof=1) / np.sqrt(300)
    assert abs(sim["objective"].mean() - exact["objective"]) < 4 * sem
    assert abs(sim["service_level"].mean() - exact["service_level"]) < 0.01


def test_nstep_target():
    # n = 1 is one-step TD
    assert abs(nstep_target([5.0], 0, 1, 0.9, 10.0) - (5.0 + 0.9 * 10.0)) < 1e-12
    # mid-episode window: R1 + R2 + gamma^2 * Q
    assert abs(nstep_target([1.0] * 5, 0, 2, 1.0, 10.0) - 12.0) < 1e-12
    # discounting restarts at the window start
    assert abs(nstep_target([10, 10, 4, 8, 10], 2, 2, 0.5, 0.0) - (4 + 0.5 * 8)) < 1e-12
    # window cut by the time limit bootstraps after the rewards that exist
    assert abs(nstep_target([1.0] * 5, 3, 4, 0.5, 8.0) - (1 + 0.5 + 0.25 * 8)) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
