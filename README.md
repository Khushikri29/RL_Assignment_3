# CS6109 – Reinforcement Learning – Assignment 3
**Passive ADP and Tabular Q-Learning (implemented from scratch with Python + NumPy)**

- **Roll No.:** 2611AI49  
- **Name:** `<your name>`
- **Course:** CS6109 – Reinforcement Learning

No built-in RL agents, Q-learning implementations or policy-learning functions are used. Only **NumPy** is used for the algorithms; **matplotlib** is used only for plotting.

---

## 1. Repository contents

| File | Description |
|---|---|
| `2611AI49_Assignment3.ipynb` | Notebook with all five questions, outputs and plots already included |
| `rl_assignment3.py` | Same code as a single runnable script (`q1()` … `q5()`) |
| `output.txt` | Full printed output of one complete run of the script |
| `q3_success_rate.png` | Q3 – success rate vs. episodes for ε ∈ {0, 0.1, 0.3, 0.5} (seed 42) |
| `q3_success_rate_20seeds.png` | Q3 – same experiment averaged over 20 random seeds |
| `q5_decay_vs_fixed.png` | Q5 – fixed ε = 0.2 vs decaying ε, plus the ε schedule |

## 2. How to run

```bash
pip install numpy matplotlib
python rl_assignment3.py > rerun.txt
diff --strip-trailing-cr rerun.txt output.txt && echo SAME
```

- The script prints the results of all five questions and saves the three PNG plots in the current folder.
- Runtime is a few seconds (it includes a 20-seed robustness check).
- All randomness is seeded (`SEED = 42`), so the printed numbers are reproducible. A different NumPy version may change the random stream slightly; the trends stay the same.
- **Colab / Jupyter:** open `2611AI49_Assignment3.ipynb` and use *Run all*. Plots appear inline.

## 3. Environment used in Q3–Q5 (deterministic 4×4 Frozen Lake)

```
S F F F        0  1  2  3
F H F H        4  5  6  7
F F F H        8  9 10 11
H F F G       12 13 14 15
```

- States 0–15 numbered row-wise. Start = 0, Goal = 15, Holes = 5, 7, 11, 12.
- Actions: 0 = Left, 1 = Down, 2 = Right, 3 = Up. An invalid move (into a wall) keeps the agent in the same state.
- Holes and Goal are terminal. Reward = **+1** on reaching the Goal, **0** otherwise.
- Q-table: 16 × 4, initialised with zeros. α = 0.5, γ = 0.9 everywhere. Max 100 steps per episode.
- An episode counts as **successful** if it ends in the Goal state.

## 4. Questions, approach and results

### Q1 – Passive ADP: model and utility update
- Count-based model: `P(s'|s,a) = N(s,a,s') / N(s,a)`.
- Initial counts `N(s,a) = 5`; outcomes of the policy action: intended direction 3 times, each perpendicular direction once. For `(s11, Down)`: → s21 ×3, → s12 ×1, → s11 (wall bump) ×1. For `(s21, Right)`: → s22 ×3, → s11 ×1, → s21 (wall bump) ×1.
- After observing ⟨s11, Down, s21, −0.04⟩: `N(s11,Down)=6`, `N(s11,Down,s21)=4`, `P(s21|s11,Down)=4/6=0.6667`.
- Policy evaluation by solving the Bellman equations `V(s) = R(s) + γ Σ P(s'|s,π(s)) V(s')` as a linear system (terminal utilities V(s12)=+1, V(s22)=−1):
  - **V(s11) = −0.4377**, **V(s21) = −0.8034** (verified by iterative evaluation).

### Q2 – TD / Q-learning update
- `TD_Target = r + γ max_a' Q(s',a')` (or `r` if s' is terminal), `TD_Error = TD_Target − Q(s,a)`, `Q(s,a) ← Q(s,a) + α·TD_Error`.
- Six sample transitions are printed with current Q, TD target, TD error and updated Q, including terminal (goal / hole) and bootstrapped transitions, and a repeated transition to show Q moving towards its target.

### Q3 – Exploration vs. exploitation (ε ∈ {0, 0.1, 0.3, 0.5}, 2000 episodes)

| ε | Successful episodes | Final success rate (last 100) | Exploratory actions |
|---|---|---|---|
| 0.0 (random tie-break) | 1900 / 2000 | 100 % | 0 |
| 0.1 | 1712 / 2000 | 91 % | 1242 |
| 0.3 | 1250 / 2000 | 64 % | 4197 |
| 0.5 | 750 / 2000 | 46 % | 7191 |
| 0.0 (plain argmax, ties → Left) | **0 / 2000** | **0 %** | 0 |

- Learned greedy policies for every ε are printed in `output.txt`; all ε that learn reach the goal.
- **Why a purely greedy agent can fail:** Q starts at zero and the only reward is at the goal, so every action ties. With plain argmax the agent always picks Left, stays in state 0, never sees a reward, and Q never changes. With random tie-breaking, ε = 0 succeeds here only because the tie-breaking is itself a hidden form of exploration.
- Larger ε explores more but its on-policy training success drops, because random moves often fall into holes.

### Q4 – Tabular Q-learning on Frozen Lake (α = 0.5, γ = 0.9, ε = 0.2, 2000 episodes)
- Final Q-table and greedy policy are printed in `output.txt`.
- Learned policy (arrows; H hole, G goal):
  ```
  v > v <
  v H v H
  > > v H
  H > > G
  ```
- Greedy path: 0 → 4 → 8 → 9 → 10 → 14 → 15 (6 steps, optimal). `max_a Q(0,a) = 0.5905 = γ⁵`, the theoretical optimal value.

### Q5 – Decaying exploration (ε₀ = 1.0, ε_min = 0.05, decay = 0.995)
- `ε_{t+1} = max(ε_min, ε_t · decay)`, updated once per episode; ε reaches 0.05 at episode 598.

| Agent | Successful episodes | Final success rate (last 100) | Exploratory actions | Greedy policy reaches goal |
|---|---|---|---|---|
| Fixed ε = 0.2 | 1480 / 2000 | 77 % | 2643 | Yes (6 steps) |
| Decaying ε | 1706 / 2000 | 96 % | 1908 | Yes (6 steps) |

- Averaged over 20 seeds: final-window success 78.1 % (fixed) vs 94.8 % (decaying); exploratory actions 2640 vs 1772; the greedy policy reaches the goal in 20/20 runs for both.
- **Effect of decreasing exploration:** explore heavily early to discover the goal and spread the +1 reward backwards; then exploit. Fixed ε keeps taking ~20 % random moves forever, so it keeps falling into holes even with a good Q-table.

## 5. Design choices and assumptions

- **Q1 model:** the 3-1-1 outcome counts are applied to both `(s11, Down)` and `(s21, Right)`; terminal-state utilities equal their rewards (+1 / −1).
- **Tie-breaking:** greedy ties are broken uniformly at random (standard practice); the plain-argmax run is reported separately for ε = 0 to demonstrate the failure case.
- **Exploratory action:** an action chosen by the random branch of ε-greedy (counted even if it happens to equal the greedy action).
- **Success windows:** success rate plots use non-overlapping windows of 100 episodes.
- **Same conditions:** every agent in Q3/Q5 uses the same environment, number of episodes, α, γ and a zero-initialised Q-table.

## 6. Concepts covered (for evaluation)
States, actions, rewards, policy, transition probability, Bellman updates, utility values, Q-values, TD target, TD error, learning rate α, discount factor γ, ε-greedy exploration, and the difference between model-based ADP and model-free Q-learning.
