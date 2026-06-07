"""
config.py - Central configuration for the Qwen3-ASR Darija fine-tuning project.

Edit this file to change dataset sizes, hyperparameters, or paths.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = "data"
TRAIN_WAV_DIR = os.path.join(DATA_DIR, "wavs", "train")
TEST_WAV_DIR = os.path.join(DATA_DIR, "wavs", "test")
TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
TEST_JSONL = os.path.join(DATA_DIR, "test.jsonl")
OUTPUT_DIR = "model_output"
EVAL_DETAILS_PATH = os.path.join(DATA_DIR, "eval_details.tsv")

QWEN_REPO_DIR = "Qwen3-ASR"
SFT_SCRIPT = os.path.join(QWEN_REPO_DIR, "finetuning", "qwen3_asr_sft.py")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_NAME = "atlasia/MoulSot-Full-80"
NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 200
RANDOM_SEED = 42

# Column name candidates for the transcript field (tried in order)
TEXT_COL_CANDIDATES = ["text", "transcription", "sentence", "transcript"]

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE = 16000   # Hz — required by Qwen3-ASR
MIN_DURATION = 0.5           # seconds — clips shorter than this are skipped
MAX_DURATION = 30.0          # seconds — clips longer than this are skipped

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3-ASR-0.6B"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUMULATION = 2       # effective batch = BATCH_SIZE * GRAD_ACCUMULATION
LEARNING_RATE = 1e-7
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
NUM_WORKERS = 1             # dataloader workers

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EVAL_SAMPLES = 50           # number of test samples to evaluate
