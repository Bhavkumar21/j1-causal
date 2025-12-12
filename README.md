# j1-nano-causal: Causal Sensitivity Training for Reward Models

## The Hypothesis

**Standard j1-nano training** uses two rewards:
- `R_format`: Rewards correct XML structure
- `R_argmax`: Rewards correct preference prediction

**The problem**: Nothing forces the model to generate criteria that ACTUALLY AFFECT the scores. The criteria can be "decorative" - they sound smart but are ignored.

**Our solution**: Add a third reward that measures causal sensitivity:
- `R_causal`: Rewards criteria where `d(score)/d(criteria) != 0`

## The Math

```
Standard j1-nano:
    R_total = R_format + R_argmax

j1-nano-causal (ours):
    R_total = R_format + R_argmax + λ * R_causal

Where:
    R_causal = || d(score) / d(criteria_embedding) ||

    High R_causal = criteria influence the scores (good!)
    Low R_causal = criteria are ignored (bad!)
```

## Visual Comparison

```
Standard j1-nano:
┌─────────────────────────────────────────────┐
│   Query ──→ Criteria ──→ Analysis ──→ Score │
│                │              (no link)  │  │
│                └─────────────────────────┘  │
│   Criteria can be ignored!                  │
└─────────────────────────────────────────────┘

j1-nano-causal (ours):
┌─────────────────────────────────────────────┐
│   Query ──→ Criteria ──→ Analysis ──→ Score │
│                │                         ↑  │
│                └──── MUST AFFECT ────────┘  │
│   R_causal enforces this connection!        │
└─────────────────────────────────────────────┘
```

## Training

### Requirements
```bash
pip install -r requirements.txt
```

### Run Training
```bash
python train_judge.py
```

## Evaluation

```bash
# Start vLLM server with LoRA
vllm serve Qwen/Qwen3-0.6B --enable-lora --lora-modules j1-nano-causal=./j1-nano-causal-lora

# Run RewardBench evaluation
python test_j1.py --model-name j1-nano-causal
```

## Key Files

- `config.py` - Configuration (model, hyperparameters, R_causal weight)
- `rewards.py` - Reward functions including new `causal_reward_func`
- `train_judge.py` - Main training script
- `test_j1.py` - RewardBench evaluation
- `utils.py` - Data loading and prompt formatting

## How R_causal Works

1. **Extract criteria** from generated output
2. **Encode criteria** to embedding vector
3. **Measure influence** via:
   - Semantic coherence between criteria and analysis
   - Score differentiation (are scores different?)
   - Criteria specificity (how detailed?)
4. **Return sensitivity** as reward

This approximates the true gradient `||d(score)/d(criteria)||` without
requiring backpropagation through the full generation.

