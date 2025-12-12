"""
Training script for j1-nano-causal: j1-nano with Causal Sensitivity Reward

Key Innovation:
    R_total = R_format + R_argmax + lambda * R_causal

Where R_causal encourages criteria to ACTUALLY AFFECT the scores:
    || d(score) / d(criteria) || != 0

This prevents the model from generating "decorative" criteria that
sound good but are ignored when scoring.

Usage:
    # Quick test (50 steps, ~30 mins)
    python train_judge.py

    # Full training (set QUICK_TEST=False in config.py)
    python train_judge.py
"""

import os
import torch
import wandb
from config import Config
from peft import LoraConfig
from rewards import format_reward_func, argmax_reward_func, causal_reward_func
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import GRPOConfig, GRPOTrainer
from utils import get_skywork_dataset

# =============================================================================
# SETUP
# =============================================================================

print("=" * 70)
print("j1-nano-causal: Training with Causal Sensitivity Reward")
print("=" * 70)
print(f"\nModel: {Config.MODEL_NAME}")
print(f"Causal reward weight (lambda): {Config.CAUSAL_REWARD_WEIGHT}")
print(f"Quick test mode: {Config.QUICK_TEST}")
if Config.QUICK_TEST:
    print(f"Quick test steps: {Config.QUICK_TEST_STEPS}")
print(f"Output: {Config.OUTPUT_DIR}")
print("=" * 70)

# Initialize wandb
wandb.init(
    project=Config.WANDB_PROJECT,
    name=Config.RUN_NAME + ("-quicktest" if Config.QUICK_TEST else ""),
    config={
        "model": Config.MODEL_NAME,
        "epochs": Config.EPOCHS,
        "batch_size": Config.BATCH_SIZE,
        "learning_rate": Config.LEARNING_RATE,
        "lora_rank": Config.LORA_RANK,
        "causal_reward_weight": Config.CAUSAL_REWARD_WEIGHT,
        "quick_test": Config.QUICK_TEST,
        "quick_test_steps": Config.QUICK_TEST_STEPS if Config.QUICK_TEST else None,
    },
    tags=["j1-nano-causal", "causal-sensitivity", "reward-model"]
)

# Log the experiment description
wandb.run.notes = """
j1-nano-causal: Testing causal sensitivity hypothesis

Hypothesis: Adding R_causal = ||d(score)/d(criteria)|| to the reward
will force the model to generate criteria that actually affect scores,
leading to better RewardBench performance.

Key metrics to watch:
- rewards/R_causal_mean: Should INCREASE over training
- causal/coherence_mean: Criteria-analysis similarity
- causal/score_diff_mean: Score differentiation
- rewards/R_argmax_mean: Prediction accuracy (main metric)
"""

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading datasets...")
train_dataset = get_skywork_dataset(Config.TRAIN_INPUT_FILE, split="train")
test_dataset = get_skywork_dataset(Config.TEST_INPUT_FILE, split="test")

# For quick test, use subset
if Config.QUICK_TEST:
    # Calculate how many samples we need for desired steps
    samples_needed = Config.QUICK_TEST_STEPS * Config.BATCH_SIZE * Config.GRAD_ACCUM
    train_dataset = train_dataset.select(range(min(samples_needed * 2, len(train_dataset))))
    print(f"Quick test mode: using {len(train_dataset)} train samples")

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# =============================================================================
# MODEL SETUP
# =============================================================================

# LoRA configuration
lora_config = LoraConfig(
    r=Config.LORA_RANK,
    lora_alpha=Config.LORA_ALPHA,
    lora_dropout=Config.LORA_DROPOUT,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

# Load model
print(f"\nLoading model: {Config.MODEL_NAME}")

# Try flash attention, fall back to sdpa or eager
try:
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_state_dict=True,
        attn_implementation="flash_attention_2",
    )
    print("Using Flash Attention 2")
except ImportError:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_state_dict=True,
            attn_implementation="sdpa",  # PyTorch native scaled dot product attention
        )
        print("Using SDPA (PyTorch native attention)")
    except:
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_state_dict=True,
        )
        print("Using default attention")
model.enable_input_require_grads()
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# TRAINING CONFIG
# =============================================================================

# Adjust settings for quick test
max_steps = Config.QUICK_TEST_STEPS if Config.QUICK_TEST else -1
save_steps = Config.SAVE_STEPS if Config.QUICK_TEST else 100
num_epochs = 1 if Config.QUICK_TEST else Config.EPOCHS

training_args = GRPOConfig(
    run_name=Config.RUN_NAME,
    output_dir=Config.OUTPUT_DIR,
    use_vllm=Config.USE_VLLM,
    # Note: vllm_device removed in TRL 0.26.0 - uses automatic device placement
    vllm_gpu_memory_utilization=Config.VLLM_USAGE,
    num_generations=Config.GROUP_NUM,
    max_prompt_length=Config.MAX_PROMPT_LENGTH,
    max_completion_length=Config.MAX_SEQ_LENGTH - Config.MAX_PROMPT_LENGTH,
    vllm_max_model_length=Config.MAX_SEQ_LENGTH,  # Fixed: was vllm_max_model_len
    per_device_train_batch_size=Config.BATCH_SIZE * Config.GROUP_NUM,
    gradient_accumulation_steps=Config.GRAD_ACCUM,
    num_train_epochs=num_epochs,
    max_steps=max_steps,
    learning_rate=Config.LEARNING_RATE,
    eval_strategy=Config.EVAL_STRATEGY,
    eval_steps=Config.EVAL_STEPS,
    gradient_checkpointing=True,
    beta=0.0005,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_steps=5,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=True,
    save_steps=save_steps,
    max_grad_norm=0.1,
    report_to="wandb",
    log_completions=True,
    overwrite_output_dir=True,
)

# =============================================================================
# CREATE TRAINER
# =============================================================================

print("\nInitializing trainer with rewards:")
print("  - R_format: XML structure validation [0, 0.2]")
print("  - R_argmax: Preference prediction accuracy [0, 1]")
print(f"  - R_causal: Causal sensitivity (NEW!) [0, {Config.CAUSAL_REWARD_WEIGHT}]")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    peft_config=lora_config,
    reward_funcs=[
        format_reward_func,   # R_format: correct XML structure
        argmax_reward_func,   # R_argmax: correct preference
        causal_reward_func,   # R_causal: causal sensitivity (NEW!)
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# =============================================================================
# TRAIN!
# =============================================================================

print("\n" + "=" * 70)
if Config.QUICK_TEST:
    print(f"Starting QUICK TEST ({Config.QUICK_TEST_STEPS} steps)...")
    print("Watch wandb for: rewards/R_causal_mean (should increase)")
else:
    print("Starting FULL training...")
print("=" * 70)

try:
    # Resume from checkpoint if it exists
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, "checkpoint-50")
    if os.path.exists(checkpoint_path) and not Config.QUICK_TEST:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()
except KeyboardInterrupt:
    print("\n" + "=" * 70)
    print("Training interrupted! Saving checkpoint...")
    print("=" * 70)

# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 70)
print(f"Saving LoRA weights to {Config.OUTPUT_DIR}")
print("=" * 70)

try:
    model.save_lora(Config.OUTPUT_DIR)
except:
    # Fallback save method
    model.save_pretrained(Config.OUTPUT_DIR)

# Log final artifact to wandb
wandb.save(os.path.join(Config.OUTPUT_DIR, "*"))

print("\nTraining complete!")
print(f"Model saved to: {Config.OUTPUT_DIR}")
print(f"\nCheck wandb dashboard: {wandb.run.url}")

if Config.QUICK_TEST:
    print("\n" + "=" * 70)
    print("QUICK TEST ANALYSIS")
    print("=" * 70)
    print("""
What to look for in wandb:

1. rewards/R_causal_mean: Is it increasing?
   - If YES: Model is learning to generate causal criteria!
   - If NO/FLAT: May need to tune lambda or check reward function

2. causal/coherence_mean: Criteria-analysis semantic similarity
   - Should be > 0.5 for good alignment

3. causal/score_diff_mean: Score differentiation
   - Should be > 0.1 (not giving same score to both)

4. rewards/R_argmax_mean: Prediction accuracy
   - Main metric, should trend upward

If R_causal is working, set QUICK_TEST=False in config.py and run full training!
""")

wandb.finish()
