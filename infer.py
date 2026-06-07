"""
infer.py - Transcribe new audio files using a fine-tuned Qwen3-ASR checkpoint.

Usage:
    python infer.py path/to/audio.wav
    python infer.py path/to/audio.wav --checkpoint path/to/checkpoint-XXX
    python infer.py path/to/audio1.wav path/to/audio2.wav
"""

import argparse
import glob
import os

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
    checkpoints = sorted(
        glob.glob(os.path.join(config.OUTPUT_DIR, "checkpoint-*")),
        key=os.path.getmtime,
    )
    if checkpoints:
        return checkpoints[-1]
    print("No checkpoint found. Using base model.")
    return config.BASE_MODEL


def patch_checkpoint_with_processor(checkpoint_path: str) -> None:
    if checkpoint_path == config.BASE_MODEL:
        return
    processor = AutoProcessor.from_pretrained(config.BASE_MODEL)
    processor.save_pretrained(checkpoint_path)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file. Converts stereo to mono if needed."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def transcribe_files(model, audio_paths: list[str]) -> None:
    for path in audio_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        audio, sr = load_audio(path)
        duration = len(audio) / sr

        results = model.transcribe(
            audio=(audio, sr),
            language="Arabic",
        )
        text = getattr(results[0], "text", "") if results else ""

        print(f"File     : {path}")
        print(f"Duration : {duration:.2f}s")
        print(f"Transcript: {text}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio files with Qwen3-ASR.")
    parser.add_argument("audio_files", nargs="+", help="Path(s) to WAV file(s) to transcribe.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint directory. Defaults to latest checkpoint in OUTPUT_DIR.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    print(f"Using model: {checkpoint_path}\n")

    patch_checkpoint_with_processor(checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model on {device} ...")
    model = Qwen3ASRModel.from_pretrained(
        checkpoint_path,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=16,
    )
    print("Model ready.\n")

    transcribe_files(model, args.audio_files)


if __name__ == "__main__":
    main()
