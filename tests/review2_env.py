"""RetailRestockEnv exactly as used in Review II, kept only as a regression
reference: tests check that the current environment's "proposal" reward
reproduces it step for step."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RetailRestockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, grid_size=4, n_shelves=2, stock_max=3, carry_capacity=2,
                 max_steps=200, demand_prob=0.12, render_mode=None, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_shelves = n_shelves
        self.stock_max = stock_max
        self.carry_capacity = carry_capacity
        self.max_steps = max_steps
        self.demand_prob = demand_prob
        self.render_mode = render_mode

        self.R_RESTOCK = 10.0
        self.R_STOCKOUT = -8.0
        self.EPS_STEP = -0.2
        self.COST_DIST = -0.1
        self.SHAPING = 2.0

        self.depot_pos = (0, 0)
        rng = np.random.default_rng(seed)
        candidates = [(x, y) for x in range(grid_size) for y in range(grid_size)
                      if (x, y) != self.depot_pos]
        chosen_idx = rng.choice(len(candidates), size=n_shelves, replace=False)
        self.shelf_positions = [candidates[i] for i in chosen_idx]

        self.action_space = spaces.Discrete(7)
        low = np.array([0, 0, 0] + [0] * n_shelves + [0] * n_shelves + [0], dtype=np.float32)
        high = np.array([grid_size - 1, grid_size - 1, carry_capacity]
                         + [stock_max] * n_shelves
                         + [20] * n_shelves
                         + [1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._np_random = rng
        self.window = None
        self.clock = None
        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.robot_pos = list(self.depot_pos)
        self.carried = self.carry_capacity
        self.stock = np.full(self.n_shelves, self.stock_max, dtype=np.int32)
        self.time_since_visit = np.zeros(self.n_shelves, dtype=np.int32)
        self.t = 0
        self.total_restocks = 0
        self.total_stockout_steps = 0
        self.total_distance = 0
        return self._get_obs(), {}

    def _get_obs(self):
        tod = (self.t % 24) / 24.0
        return np.array(
            [self.robot_pos[0], self.robot_pos[1], self.carried]
            + list(self.stock)
            + list(np.clip(self.time_since_visit, 0, 20))
            + [tod],
            dtype=np.float32,
        )

    def discretize(self, obs):
        x, y, carried = int(obs[0]), int(obs[1]), int(obs[2])
        stocks = tuple(int(s) for s in obs[3:3 + self.n_shelves])
        tod_bin = int(obs[-1] * 2)
        return (x, y, carried, stocks, tod_bin)

    def step(self, action):
        self.t += 1
        reward = self.EPS_STEP
        prev_pos = tuple(self.robot_pos)

        moved = False
        if action == 0 and self.robot_pos[1] > 0:
            self.robot_pos[1] -= 1; moved = True
        elif action == 1 and self.robot_pos[1] < self.grid_size - 1:
            self.robot_pos[1] += 1; moved = True
        elif action == 2 and self.robot_pos[0] > 0:
            self.robot_pos[0] -= 1; moved = True
        elif action == 3 and self.robot_pos[0] < self.grid_size - 1:
            self.robot_pos[0] += 1; moved = True
        elif action == 4:
            reward += self._do_restock()
        elif action == 5:
            if tuple(self.robot_pos) == self.depot_pos:
                self.carried = self.carry_capacity
            else:
                reward += -1.0

        if moved:
            dist = 1
            self.total_distance += dist
            reward += self.COST_DIST * dist

        low_idx = int(np.argmin(self.stock))
        target = self.shelf_positions[low_idx]
        prev_d = abs(prev_pos[0] - target[0]) + abs(prev_pos[1] - target[1])
        new_d = abs(self.robot_pos[0] - target[0]) + abs(self.robot_pos[1] - target[1])
        if new_d < prev_d:
            reward += self.SHAPING

        for i in range(self.n_shelves):
            self.time_since_visit[i] += 1
            if self._np_random.random() < self.demand_prob and self.stock[i] > 0:
                self.stock[i] -= 1

        stockouts_now = int(np.sum(self.stock == 0))
        if stockouts_now > 0:
            reward += self.R_STOCKOUT * stockouts_now
            self.total_stockout_steps += stockouts_now

        terminated = False
        truncated = self.t >= self.max_steps
        info = {
            "restocks": self.total_restocks,
            "stockout_steps": self.total_stockout_steps,
            "distance": self.total_distance,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _do_restock(self):
        pos = tuple(self.robot_pos)
        if pos in self.shelf_positions and self.carried > 0:
            idx = self.shelf_positions.index(pos)
            if self.stock[idx] < self.stock_max:
                was_before_stockout = self.stock[idx] > 0
                units = min(self.carry_capacity, self.stock_max - self.stock[idx], self.carried)
                if units > 0:
                    self.stock[idx] += units
                    self.carried -= units
                    self.time_since_visit[idx] = 0
                    self.total_restocks += 1
                    return self.R_RESTOCK if was_before_stockout else self.R_RESTOCK * 0.5
        return -1.0

    def close(self):
        if self.window is not None:
            import pygame
            pygame.quit()
            self.window = None