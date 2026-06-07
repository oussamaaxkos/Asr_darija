# Qwen3-ASR Fine-Tuning for Moroccan Darija

Fine-tune [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) on the [MoulSot-Full](https://huggingface.co/datasets/atlasia/MoulSot-Full) dataset — a curated corpus of Moroccan Darija speech.

Based on the [MoulSot project](https://huggingface.co/blog/abdeljalilELmajjodi/moulsot), which built a full pipeline crawling ~1,500 hours from YouTube and transcribing 80 hours using Gemini 2.5 Pro.

---

## Project Structure

```
qwen3_asr_darija/
├── config.py          # All paths and hyperparameters
├── check_env.py       # Verify GPU and installed packages
├── prepare_data.py    # Download dataset and build JSONL manifests
├── train.py           # Fine-tune using the official Qwen3-ASR SFT script
├── evaluate.py        # Evaluate a checkpoint with WER / CER / RTF
├── infer.py           # Transcribe new audio files
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (16 GB VRAM recommended; T4 works for the default config)
- Git

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Check your environment

```bash
python check_env.py
```

### 2. Prepare the data

Downloads MoulSot-Full from Hugging Face, resamples audio to 16 kHz, and writes JSONL manifests.

```bash
python prepare_data.py
```

By default this uses **1,000 training** and **200 test** samples. Edit `config.py` to change these values.

### 3. Fine-tune the model

```bash
python train.py
```

This will:
1. Clone the official `QwenLM/Qwen3-ASR` repository.
2. Apply the fp16 compatibility patch.
3. Launch the SFT training script.

Checkpoints are saved to `model_output/`.

### 4. Evaluate

```bash
python evaluate.py
```

Optional arguments:

```bash
python evaluate.py --checkpoint model_output/checkpoint-250 --samples 100
```

Reports WER, CER, and mean RTF. Per-sample results are saved to `data/eval_details.tsv`.

### 5. Transcribe new audio

```bash
python infer.py path/to/audio.wav
python infer.py audio1.wav audio2.wav --checkpoint model_output/checkpoint-250
```

---

## Configuration

All settings live in `config.py`. Common things to change:

| Setting | Default | Description |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen3-ASR-0.6B` | Pre-trained model to fine-tune |
| `NUM_TRAIN_SAMPLES` | `1000` | Training samples (more = better results, slower) |
| `EPOCHS` | `3` | Training epochs |
| `BATCH_SIZE` | `4` | Per-GPU batch size |
| `LEARNING_RATE` | `1e-7` | Learning rate |
| `EVAL_SAMPLES` | `50` | Samples used during evaluation |

---

## Data Format

Qwen3-ASR expects JSONL files where each line has the format:

```json
{"audio": "/absolute/path/to/clip.wav", "text": "language Arabic<asr_text>النص هنا"}
```

- Audio must be **16 kHz mono WAV**.
- The `language Arabic<asr_text>` prefix is required — it is how the model identifies the target language.

`prepare_data.py` handles all of this automatically.

---

## Metrics

| Metric | Description |
|---|---|
| **WER** | Word Error Rate — lower is better; 0.0 = perfect |
| **CER** | Character Error Rate — often more meaningful for Arabic script |
| **RTF** | Real-Time Factor — RTF < 1.0 means faster than real-time |

---

## Background

**Moroccan Darija** is spoken by 30+ million people but is severely under-resourced in speech technology. Unlike Modern Standard Arabic, Darija blends Arabic roots with Amazigh, French, and Spanish influences.

The MoulSot project addressed this by building a clean 80-hour corpus and fine-tuning Qwen3-ASR — achieving strong WER on a dialect where almost no prior ASR work existed.

---

## References

- [Qwen3-ASR Model Card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [MoulSot Dataset](https://huggingface.co/datasets/atlasia/MoulSot-Full)
- [MoulSot Fine-tuned Model](https://huggingface.co/atlasia/moulsot.v0.3)
- [MoulSot Blog Post](https://huggingface.co/blog/abdeljalilELmajjodi/moulsot)
- [Live Demo](https://huggingface.co/spaces/atlasia/MoulSot.v0.3)
