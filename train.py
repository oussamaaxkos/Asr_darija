"""
train.py - Fine-tune Qwen3-ASR on Moroccan Darija using the official SFT script.

Steps performed:
  1. Clone the Qwen3-ASR repository (if not already present).
  2. Apply the fp16 patch required on Colab / single-GPU setups.
  3. Launch training via torchrun.

Usage:
    python train.py
"""

import os
import subprocess
import sys

import config


# ---------------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------------

def clone_repo() -> None:
    """Clone the official Qwen3-ASR repository if it does not exist."""
    if not os.path.exists(config.QWEN_REPO_DIR):
        print("Cloning Qwen3-ASR repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/QwenLM/Qwen3-ASR.git"],
            check=True,
        )
        print("Repository cloned.")
    else:
        print("Qwen3-ASR repository already exists, skipping clone.")

    req_path = os.path.join(config.QWEN_REPO_DIR, "finetuning", "requirements.txt")
    if os.path.exists(req_path):
        print("Installing fine-tuning requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req_path], check=True)
        print("Requirements installed.")


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

def apply_fp16_patch() -> None:
    """
    Patch the SFT script to disable fp16 training.

    The official script sets fp16=not use_bf16, which conflicts with
    bf16 on single-GPU setups. This fix forces fp16=False.
    """
    if not os.path.exists(config.SFT_SCRIPT):
        raise FileNotFoundError(
            f"SFT script not found at {config.SFT_SCRIPT}. "
            "Run this script again after the repository is cloned."
        )

    with open(config.SFT_SCRIPT, "r") as f:
        code = f.read()

    patched = code.replace("fp16=not use_bf16,", "fp16=False,")

    if patched == code:
        print("fp16 patch already applied or not needed.")
    else:
        with open(config.SFT_SCRIPT, "w") as f:
            f.write(patched)
        print("fp16 patch applied.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_train_command() -> list[str]:
    """Build the torchrun command for fine-tuning."""
    train_jsonl = os.path.abspath(config.TRAIN_JSONL)
    test_jsonl = os.path.abspath(config.TEST_JSONL)
    output_dir = os.path.abspath(config.OUTPUT_DIR)

    cmd = [
        "python", config.SFT_SCRIPT,
        "--model_path",          config.BASE_MODEL,
        "--train_file",          train_jsonl,
        "--eval_file",           test_jsonl,
        "--output_dir",          output_dir,
        "--epochs",              str(config.EPOCHS),
        "--batch_size",          str(config.BATCH_SIZE),
        "--grad_acc",            str(config.GRAD_ACCUMULATION),
        "--lr",                  str(config.LEARNING_RATE),
        "--warmup_ratio",        str(config.WARMUP_RATIO),
        "--lr_scheduler_type",   config.LR_SCHEDULER,
        "--num_workers",         str(config.NUM_WORKERS),
    ]
    return cmd


def verify_data_files() -> None:
    """Ensure the JSONL manifests exist before training."""
    for path in [config.TRAIN_JSONL, config.TEST_JSONL]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                "Run prepare_data.py first."
            )


def check_checkpoints() -> None:
    """List saved checkpoints after training."""
    import glob
    checkpoints = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "checkpoint-*")))
    if checkpoints:
        print(f"\nLatest checkpoint: {checkpoints[-1]}")
    else:
        print("\nNo checkpoints found. Check training logs for errors.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("Qwen3-ASR Fine-Tuning — Moroccan Darija")
    print("=" * 50)

    # 1. Setup repository
    clone_repo()

    # 2. Patch the SFT script
    apply_fp16_patch()

    # 3. Verify data is ready
    verify_data_files()

    # 4. Build and print the training command
    cmd = build_train_command()
    print("\nLaunching training with command:")
    print("  " + " ".join(cmd))
    print()

    # 5. Run training
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nTraining exited with code {result.returncode}.")
        sys.exit(result.returncode)

    print("\nTraining complete.")
    check_checkpoints()


if __name__ == "__main__":
    main()
