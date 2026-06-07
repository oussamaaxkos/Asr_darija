"""
prepare_data.py - Download the MoulSot dataset and prepare JSONL manifests
                  with 16 kHz WAV files for Qwen3-ASR fine-tuning.

Usage:
    python prepare_data.py
"""

import json
import os

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_text_column(features) -> str:
    """Return the name of the transcript column in the dataset."""
    for candidate in config.TEXT_COL_CANDIDATES:
        if candidate in features:
            return candidate
    raise ValueError(
        f"Cannot find a transcript column. Available columns: {list(features.keys())}"
    )


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = config.TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample audio to target_sr if needed."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def prepare_split(dataset, split_name: str, wav_dir: str, text_col: str) -> str:
    """
    Convert a HuggingFace dataset split into WAV files + JSONL manifest.

    Args:
        dataset:    HuggingFace Dataset object.
        split_name: 'train' or 'test'.
        wav_dir:    Directory where WAV files will be saved.
        text_col:   Name of the transcript column.

    Returns:
        Path to the generated .jsonl file.
    """
    os.makedirs(wav_dir, exist_ok=True)
    jsonl_path = os.path.join(config.DATA_DIR, f"{split_name}.jsonl")

    written = 0
    skipped = 0

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for idx, example in enumerate(tqdm(dataset, desc=f"Preparing {split_name}")):

            # --- Transcript ---
            try:
                transcript = example.get(text_col, "").strip()
            except Exception:
                skipped += 1
                continue

            # --- Audio ---
            audio_info = example["audio"]
            audio_array = audio_info["array"].astype(np.float32)
            orig_sr = audio_info["sampling_rate"]

            # Resample to 16 kHz
            audio_16k = resample(audio_array, orig_sr)

            # Skip clips outside the acceptable duration range
            duration = len(audio_16k) / config.TARGET_SAMPLE_RATE
            if duration < config.MIN_DURATION or duration > config.MAX_DURATION:
                skipped += 1
                continue

            # --- Save WAV ---
            wav_filename = f"{split_name}_{idx:06d}.wav"
            wav_path = os.path.abspath(os.path.join(wav_dir, wav_filename))
            sf.write(wav_path, audio_16k, config.TARGET_SAMPLE_RATE)

            # --- Write JSONL entry ---
            # Qwen3-ASR requires this exact text format:
            #   "language Arabic<asr_text>{transcript}"
            qwen_text = f"language Arabic<asr_text>{transcript}"
            entry = {"audio": wav_path, "text": qwen_text}
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"{split_name}: {written} samples written, {skipped} skipped")
    print(f"Manifest saved to: {jsonl_path}")
    return jsonl_path


def inspect_manifest(jsonl_path: str, n: int = 3) -> None:
    """Print the first n entries of a JSONL manifest."""
    print(f"\nFirst {n} lines of {jsonl_path}:")
    print("-" * 70)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            text_preview = entry["text"][:100] + "..." if len(entry["text"]) > 100 else entry["text"]
            print(f"  audio : {entry['audio']}")
            print(f"  text  : {text_preview}")
            print()
            if i + 1 >= n:
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # --- Load datasets ---
    print("Loading training split from Hugging Face...")
    train_ds_full = load_dataset(
        config.DATASET_NAME,
        split="train",
        trust_remote_code=True,
    )
    print(f"Full training set: {len(train_ds_full):,} samples")

    print("Loading test split from Hugging Face...")
    test_ds_full = load_dataset(
        config.DATASET_NAME,
        split="test",
        trust_remote_code=True,
    )
    print(f"Full test set: {len(test_ds_full):,} samples")

    # --- Subsample ---
    train_ds = train_ds_full.shuffle(seed=config.RANDOM_SEED).select(
        range(config.NUM_TRAIN_SAMPLES)
    )
    test_ds = test_ds_full.shuffle(seed=config.RANDOM_SEED).select(
        range(min(config.NUM_TEST_SAMPLES, len(test_ds_full)))
    )

    print(f"\nUsing {len(train_ds):,} training samples and {len(test_ds):,} test samples.")
    print("Adjust NUM_TRAIN_SAMPLES / NUM_TEST_SAMPLES in config.py to change this.\n")

    # --- Detect transcript column ---
    text_col = detect_text_column(train_ds.features)
    print(f"Transcript column: '{text_col}'")

    # --- Prepare splits ---
    prepare_split(train_ds, "train", config.TRAIN_WAV_DIR, text_col)
    prepare_split(test_ds, "test", config.TEST_WAV_DIR, text_col)

    # --- Sanity check ---
    inspect_manifest(config.TRAIN_JSONL)

    with open(config.TRAIN_JSONL) as f:
        n_train = sum(1 for _ in f)
    with open(config.TEST_JSONL) as f:
        n_test = sum(1 for _ in f)

    print(f"Total training samples : {n_train}")
    print(f"Total test samples     : {n_test}")


if __name__ == "__main__":
    main()
