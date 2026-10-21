# Retail shelf-restocking robot: reinforcement learning measured against the exact optimum

A robot keeps two shelves stocked in a small store under random demand. The store is small enough to solve exactly (768 Markov states), so every learned policy, whether a Q-table or a neural network, can be scored against the true optimum with no sampling noise. The project uses that to answer three questions:

1. **Does the reward we train on change what is optimal?** (exact dynamic programming)
2. **How far do nine learning algorithms get in 120,000 steps, under four training rewards?**
3. **Why do they fall short?** (by looking inside the learned values against `V*`)

BCSE432E Reinforcement Learning, VIT Chennai. Sakshi Bansal (23BPS1012), guided by Dr. D. Jeya Mala.
The full report is in [`report/main.pdf`](report/main.pdf).

## Main findings

| | |
|---|---|
| **Reward audit (exact)** | The proposal's "closer" bonus is not potential-based: stepping toward and away from a shelf pays. Its optimal policy does walk 6% further on average (up to 17%), but costs under 0.3% of the objective. Potential-based versions of the same guidance leave the optimal policy unchanged in all 768 states of all 10 layouts. |
| **Learners** | With identical budgets (120,000 steps), DQN and Double DQN come closest to the optimum: normalised score 0.96 on the proposal reward, against 0.79 for PPO and at most 0.73 for any tabular method. Both proposal bonuses help DQN (+0.12 and +0.10); their potential-based forms recover about half of that. |
| **Why** | For the tabular learners, the initial Q-value matters more than the reward. Potential-based shaping is equivalent to initialisation (Wiewiora, 2003), and starting from -150 in place of 0 triples Q-Learning's score (0.23 to 0.69) and lifts every TD learner. Counting visits shows the two failure modes either side of that value. DQN is insensitive to the initial value. REINFORCE's argmax policy scores less than half of its stochastic policy. |

## Repository layout

```
code/
  retail_env.py        Gymnasium environment; reward components reported separately
  mdp.py               exact model: transitions, VI, PI, exact episode metrics
  evaluation.py        normalised score, exact evaluation, held-out simulation
  tabular.py           Q-Learning, SARSA, MC control, 4-step Sarsa, SARSA(lambda)
  deep.py              DQN, Double DQN, REINFORCE + baseline, PPO (PyTorch)
  run_dp_audit.py      exact planning results          -> results/dp_audit.json
  run_experiments.py   every learner x reward x layout  -> results/runs/*.json
  analyze.py           tables and figures for the report -> results/tables, results/figures
  export_replays.py    replays + details for the 3D viewer -> results/replays
notebooks/             four executed notebooks that walk through the whole study
tests/test_core.py     model-vs-simulator, PBRS invariance, VI/PI, exact-vs-simulated checks
report/                IEEE-format LaTeX source and PDF
viewer/index.html      3D replay viewer (Three.js, no build step)
results/               every number in the report (JSON/CSV), and the figures
```

## Notebooks

| notebook | contents |
|---|---|
| `01_environment_and_model` | the simulator, reward components, the loop the proposal reward pays for, the exact model and its validation |
| `02_exact_planning_and_reward_audit` | VI and PI (and why they seem to disagree), the reward audit, the discount factor, the service/travel frontier, the travel-time bound |
| `03_learning_agents` | two agents trained live, the full score table with paired tests, learning curves, operational metrics, greedy vs stochastic evaluation |
| `04_why_learners_fall_short` | optimism left in the Q-tables, shaping = initialisation shown live, the initial-value study and its mechanism, DQN |

Each notebook imports the code in `code/` and is stored with its outputs.

## 3D replay viewer (demo)

`viewer/index.html` shows each agent driving the robot through the same held-out 200-step episode, in 3D, in any of the ten store layouts: the exact optimum, the nine learners (Proposal reward), and Q-Learning / SARSA(λ) started from Q₀ = −150. Shelves change colour and height with their stock, the robot carries its crates, and the HUD tracks the objective step by step. **Results & details** (or `R`) opens a panel with that agent's exact results on every layout, its update rule, configuration and the report figures.

```bash
python -m http.server 8000          # from the repository root
# open http://localhost:8000/viewer/
```

Space plays/pauses, ← → step, the slider scrubs, the speed menu goes up to 4×. URL options: `?layout=4&algo=qlearning_q150`. Three.js loads from a CDN, so the page needs internet access once. The replays are written by `python code/export_replays.py` from the policies stored in `results/runs/`; nothing is retrained.

## Reproducing the results

Python 3.11. Versions are pinned in `requirements.txt`; everything runs on a CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests -q                 # seconds: checks the model and the evaluation
cd code
python run_dp_audit.py                    # ~20 s
python run_experiments.py                 # main grid: 9 learners x 4 rewards x 10 layouts; ~40 min on 10 cores
for q in -50 -100 -150 -200 -300; do      # initial-value study (tabular), ~5 min
  python run_experiments.py --algos qlearning sarsa mc nstep_sarsa sarsa_lambda \
      --rewards objective --q-init $q --tag init$q
done
python run_experiments.py --algos dqn double_dqn --rewards objective --q-init -150 --tag init-150
python run_experiments.py --algos dqn qlearning sarsa nstep_sarsa sarsa_lambda \
    --rewards objective_restock objective_closer    # one proposal term at a time
python analyze.py                         # tables + figures
python export_replays.py                  # replays for the 3D viewer
```

Runs are seeded (layout seed = training seed), so re-running reproduces the stored JSON files. A run whose output file exists is skipped, so an interrupted grid resumes where it stopped.

## Data

There is no external dataset: all data are generated by the simulator. `results/runs/*.json` holds, for every run, the learning curve (exact score every 20 episodes), final exact metrics, the greedy policy over all 768 states, and a 100-episode simulation check on held-out demand seeds. `results/dp_audit.json` holds the exact planning results per layout.
