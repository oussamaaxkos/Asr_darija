"""
evaluate.py - Evaluate a fine-tuned Qwen3-ASR checkpoint using WER, CER, and RTF.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint path/to/checkpoint-XXX
    python evaluate.py --samples 100
"""

import argparse
import glob
import json
import os
import time

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor
from qwen_asr import Qwen3ASRModel

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint() -> str:
    """Return the path to the most recent training checkpoint, or the base model."""
    checkpoints = sorted(
        glob.glob(os.path.join(config.OUTPUT_DIR, "checkpoint-*")),
        key=os.path.getmtime,
    )
    if checkpoints:
        return checkpoints[-1]
    print("No checkpoint found in output directory. Falling back to base model.")
    return config.BASE_MODEL


def patch_checkpoint_with_processor(checkpoint_path: str) -> None:
    """
    Copy processor files from the base model into the checkpoint directory.

    The SFT script does not always save processor files alongside the weights,
    so Qwen3ASRModel.from_pretrained fails without this step.
    """
    if checkpoint_path == config.BASE_MODEL:
        return  # Base model already has processor files
    print(f"Patching processor files into {checkpoint_path} ...")
    processor = AutoProcessor.from_pretrained(config.BASE_MODEL)
    processor.save_pretrained(checkpoint_path)


def normalise(text: str) -> str:
    """Lowercase and strip whitespace for fair WER/CER comparison."""
    return str(text).strip().lower() if text else ""


def compute_metrics(refs: list[str], hyps: list[str]) -> dict:
    """Compute WER and CER, skipping empty references."""
    valid_refs, valid_hyps = [], []
    for r, h in zip(refs, hyps):
        if r.strip():
            valid_refs.append(r)
            valid_hyps.append(h)

    if not valid_refs:
        return {"wer": 1.0, "cer": 1.0}

    return {
        "wer": jiwer.wer(valid_refs, valid_hyps),
        "cer": jiwer.cer(valid_refs, valid_hyps),
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def load_test_entries(jsonl_path: str, n: int) -> list[dict]:
    """Load up to n entries from the test JSONL manifest."""
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line.strip()))
            if len(entries) >= n:
                break
    return entries


def run_evaluation(model, test_entries: list[dict], device: str) -> tuple[dict, list[dict]]:
    """
    Transcribe each test entry and compute aggregate metrics.

    Returns:
        metrics:      dict with wer, cer, mean_rtf.
        results_list: per-sample details.
    """
    references = []
    hypotheses = []
    rtf_values = []
    results_list = []

    for i, entry in enumerate(test_entries):
        wav_path = entry["audio"]
        reference = normalise(entry["text"])

        audio_array, sr = sf.read(wav_path)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)  # stereo -> mono
        duration = len(audio_array) / sr

        t0 = time.time()
        results = model.transcribe(
            audio=(audio_array.astype(np.float32), sr),
            language="Arabic",
        )
        elapsed = time.time() - t0

        raw_hyp = getattr(results[0], "text", "") if results else ""
        hypothesis = normalise(raw_hyp)
        rtf = elapsed / duration if duration > 0 else 0.0

        references.append(reference)
        hypotheses.append(hypothesis)
        rtf_values.append(rtf)
        results_list.append({
            "id":         i,
            "reference":  reference,
            "hypothesis": hypothesis,
            "duration":   round(duration, 2),
            "rtf":        round(rtf, 3),
        })

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(test_entries)} samples...")

    metrics = compute_metrics(references, hypotheses)
    metrics["mean_rtf"] = float(np.mean(rtf_values))
    return metrics, results_list


def save_results(results_list: list[dict], output_path: str) -> None:
    """Save per-sample results as a TSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("id\tduration\trtf\treference\thypothesis\n")
        for r in results_list:
            f.write(f"{r['id']}\t{r['duration']}\t{r['rtf']}\t{r['reference']}\t{r['hypothesis']}\n")
    print(f"Per-sample results saved to: {output_path}")


def print_summary(metrics: dict, n_samples: int) -> None:
    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  Samples evaluated : {n_samples}")
    print(f"  WER               : {metrics['wer']:.4f}  ({metrics['wer'] * 100:.1f}%)")
    print(f"  CER               : {metrics['cer']:.4f}  ({metrics['cer'] * 100:.1f}%)")
    print(f"  Mean RTF          : {metrics['mean_rtf']:.3f}")
    print("=" * 50)

    wer = metrics["wer"]
    if wer < 0.3:
        print("Good WER. The model has learned Darija well.")
    elif wer < 0.6:
        print("Moderate WER. More training data or epochs may improve results.")
    else:
        print("High WER — expected with limited training samples.")
        print("Try increasing NUM_TRAIN_SAMPLES in config.py.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3-ASR checkpoint on the test set.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint directory. Defaults to the latest checkpoint in OUTPUT_DIR.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=config.EVAL_SAMPLES,
        help=f"Number of test samples to evaluate (default: {config.EVAL_SAMPLES}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Determine which model to evaluate
    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    print(f"Evaluating checkpoint: {checkpoint_path}")

    # 2. Patch processor files if needed
    patch_checkpoint_with_processor(checkpoint_path)

    # 3. Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model on {device} ...")
    model = Qwen3ASRModel.from_pretrained(
        checkpoint_path,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=16,
    )
    print("Model loaded.")

    # 4. Load test data
    if not os.path.exists(config.TEST_JSONL):
        raise FileNotFoundError(
            f"Test manifest not found: {config.TEST_JSONL}\n"
            "Run prepare_data.py first."
        )
    test_entries = load_test_entries(config.TEST_JSONL, args.samples)
    print(f"Evaluating on {len(test_entries)} test samples...\n")

    # 5. Run evaluation
    metrics, results_list = run_evaluation(model, test_entries, device)

    # 6. Save and print results
    save_results(results_list, config.EVAL_DETAILS_PATH)
    print_summary(metrics, len(test_entries))


if __name__ == "__main__":
    main()
