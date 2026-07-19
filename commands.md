# Alpha Generation Pipeline: GP, PPO, and GFN

To generate thousands of alphas, you need to understand how the three main search algorithms explore the mathematical space, and how to configure them to find _different_ things.

## 1. Genetic Programming (GP)

**What it searches for:** GP treats alpha formulas like DNA (syntax trees). It starts with a massive random population of formulas and applies "evolution" (crossover: swapping parts of two formulas; mutation: randomly changing a node).
**Why use it:** It is entirely unconstrained. It searches highly chaotic, bizarre combinations that gradient-based methods would never find. It is great for finding completely out-of-the-box alpha structures.

### GP Generation Commands

Run these to brute-force different branches of the evolutionary tree. The key is to run many different seeds and vary the training years to find regime-specific alphas.

- **Train** : up to `train_end_year`
- **Valid** : `train_end_year + 1`
- **Test** : `train_end_year + 2` to `train_end_year + 4`

```bash
# Baseline run (GP automatically evaluates and saves pools of sizes 10, 20, 50, and 100, so you will get the top 50 alphas by default in the JSON output!)
pdm run python train_GP.py --seed 1 --instruments csi300 --train-end-year 2016

# Run across different random seeds to explore different evolutionary paths
for seed in {1..10}; do
    pdm run python train_GP.py --seed $seed --instruments csi300 --train-end-year 2016
done

# Train on a different market regime (e.g., end year 2014)
pdm run python train_GP.py --seed 42 --instruments csi300 --train-end-year 2014
```

## 2. Proximal Policy Optimization (PPO) - Reinforcement Learning

**What it searches for:** PPO trains a neural network (policy) to build the alpha formula token by token. It uses gradients to climb to the absolute highest reward (IC).
**Why use it:** It is incredibly greedy and efficient at finding the highest-Sharpe formulas. However, it suffers from "mode collapse"—it will often find one amazing alpha structure (e.g., a specific momentum signal) and then just generate 50 slightly tweaked versions of it.

### PPO Generation Commands

To force PPO to find different alphas, you must change the random initialization and the pool capacity (which forces it to find more than 1 alpha).

```bash
# Baseline run saving the top 50 alphas (The script will save exactly 50 alphas because of --pool 50)
pdm run python train_ppo.py --seed 0 --instruments csi300 --pool 50 --train-end-year 2016

# Increase the steps to force it to refine the 50 distinct formulas further
pdm run python train_ppo.py --seed 1 --instruments csi300 --pool 50 --steps 300000 --train-end-year 2016

# Loop across multiple seeds to avoid falling into the same local optimum, always saving top 50
for seed in {1..5}; do
    pdm run python train_ppo.py --seed $seed --instruments csi300 --pool 50 --train-end-year 2016
done
```

## 3. Generative Flow Networks (GFN - AlphaSAGE)

**What it searches for:** GFNs are designed to _sample proportionally to the reward_. Instead of climbing to the highest peak and staying there (like PPO), a GFN learns the entire map of the reward landscape and samples from all the different peaks.
**Why use it:** It explicitly searches for **diverse, high-quality portfolios**. If you want 100 alphas that are all profitable but completely uncorrelated, this is your best tool.

### GFN Generation Commands

To scale GFN, you manipulate the "Diversity vs. Exploitation" levers: `entropy_coef` (randomness), `nov_weight` (bonus for weird formulas), and `target_days` (what horizon it predicts).

```bash
# 1. High Novelty Search (forces the AI to prioritize unique, weird structures)
pdm run python train_gfn.py \
    --seed 101 \
    --pool_capacity 100 \
    --n_episodes 20000 \
    --encoder_type gnn \
    --entropy_coef 0.05 \
    --nov_weight 0.8 \
    --target_days 20

# 2. Short-term Reversion Search (Predict 1-day returns instead of 20-day)
pdm run python train_gfn.py \
    --seed 102 \
    --pool_capacity 50 \
    --target_days 1 \
    --nov_weight 0.3

# 3. High Turnover / Low Capacity (Penalize alphas that trade too often)
pdm run python train_gfn.py \
    --seed 103 \
    --pool_capacity 50 \
    --turnover_penalty_coef 0.01 \
    --target_days 5

# 4. Massive Parallel GFN Sweep
for seed in {1..5}; do
    pdm run python train_gfn.py \
        --seed $seed \
        --pool_capacity 100 \
        --n_episodes 15000 \
        --encoder_type gnn \
        --nov_weight 0.5 \
        --target_days 10
done
```

## Next Step: The Forge (Filtering & Evaluation)

Once the generation commands finish, you will have directories full of JSON files containing thousands of formulas.

**Step 1: Extract the top alphas from your runs**
You must extract the top formulas from your raw GP or PPO/GFN outputs into a standardized format.

```bash
# Extract the top 50 alphas from a GP run (Parses the cache and sorts by IC)
python extract_top_alphas.py --file out_gp/csi300_2016_day_1/40.json --is_gp --top_n 50

# Extract the top 50 alphas from a PPO or GFN run
python extract_top_alphas.py --file data/ppo_logs/pool_50/rl_model_20260719/200000_steps_pool.json --top_n 50
```

*(This will generate files named `40_top50.json` or `..._pool_top50.json`)*

**Step 2: Find the Trophy Alphas**
Run the extracted alphas through the strict trophy evaluator. This script checks if the alphas meet the high Sharpe, low turnover, and >200 stock constraints, and then ensures they have < 50% correlation with each other.

```bash
# Evaluate the extracted alphas
python evaluate_trophies.py --expressions_file out_gp/csi300_2016_day_1/40_top50.json --instruments csi300
```

*(If any survive, it will save them as `40_top50_trophy.json`)*

**Step 3: Final Combination**
Once you have collected all your surviving trophies from different seeds and methods, combine them to form a final orthogonal portfolio using AlphaForge:

```bash
python run_adaptive_combination.py --expressions_file your_trophy_folder/ --instruments csi300 --cuda 0 --train_end_year 2016
```
