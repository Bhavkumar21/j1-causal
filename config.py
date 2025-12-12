class Config:
    # VLLM settings - disabled for simplicity (0.6B model is fast enough without it)
    USE_VLLM = False
    VLLM_DEVICE = "cuda:0"
    VLLM_USAGE = 0.3  # Slightly higher for H100

    # Training settings
    EPOCHS = 10
    GROUP_NUM = 2
    BATCH_SIZE = 8  # Can increase on H100
    GRAD_ACCUM = 2
    LEARNING_RATE = 1e-4
    EVAL_STRATEGY = "no"
    EVAL_STEPS = 1000

    # Quick test mode - set to True to run just 50 steps
    QUICK_TEST = False
    QUICK_TEST_STEPS = 50  # Number of steps for quick test

    # Checkpointing - save frequently so you can stop anytime
    SAVE_STEPS = 100  # Save every 100 steps during full training

    # Model settings - Using 0.6B for j1-nano-causal
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    MAX_SEQ_LENGTH = 4096
    MAX_PROMPT_LENGTH = 2048
    LORA_RANK = 16
    LORA_ALPHA = LORA_RANK * 2
    LORA_DROPOUT = 0.1

    # Input settings
    TRAIN_INPUT_FILE = "train_df.csv"
    TEST_INPUT_FILE = "test_df.csv"

    # Output settings
    OUTPUT_DIR = "j1-nano-causal-lora"
    RUN_NAME = OUTPUT_DIR
    WANDB_PROJECT = "j1-nano-causal"

    # Column names from Skywork v2.0 dataset
    COLUMN_INPUT = "sky_input"
    COLUMN_CHOSEN_POSITION = "chosen_positions"
    COLUMN_CHOSEN = "sky_chosen"
    COLUMN_REJECTED = "sky_rejected"
    COLUMN_PROMPT = "prompt"
    COLUMN_SOURCE = "source"
    COLUMN_CHOSEN_ORIG = "chosen"
    COLUMN_REJECTED_ORIG = "rejected"

    # Chosen position
    CHOSEN_POSITION_A = "a"
    CHOSEN_POSITION_B = "b"

    # NEW: Causal reward settings
    CAUSAL_REWARD_WEIGHT = 0.1  # Lambda for R_causal
    NUM_PERTURBATIONS = 5       # Number of perturbations to estimate sensitivity
    PERTURBATION_STD = 0.1      # Standard deviation for perturbations
