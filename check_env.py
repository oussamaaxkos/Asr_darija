"""
check_env.py - Verify GPU and Python environment before training.
"""

import subprocess
import sys


def check_gpu():
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print("No GPU detected. Enable GPU runtime or use a CUDA-capable machine.")
        print("Training on CPU will be very slow.")
        return False


def check_torch():
    try:
        import torch
        print(f"PyTorch version  : {torch.__version__}")
        print(f"CUDA available   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU name         : {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory (GB)  : {mem_gb:.1f}")
        return torch.cuda.is_available()
    except ImportError:
        print("PyTorch is not installed. Run: pip install torch")
        return False


def check_packages():
    required = [
        "transformers",
        "datasets",
        "accelerate",
        "soundfile",
        "librosa",
        "jiwer",
        "tqdm",
        "huggingface_hub",
        "qwen_asr",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        return False

    print("All required packages are installed.")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)
    gpu_ok = check_gpu()
    torch_ok = check_torch()
    pkgs_ok = check_packages()

    print()
    if gpu_ok and torch_ok and pkgs_ok:
        print("Environment is ready for training.")
        sys.exit(0)
    else:
        print("Some checks failed. Address the issues above before training.")
        sys.exit(1)
