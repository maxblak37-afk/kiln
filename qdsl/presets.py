"""Профили железа и оценка VRAM на одну карту (минимумы из документации Unsloth)."""
import re

HARDWARE = {
    "rtx-3060-laptop": {
        "gpus": 1,
        "vram_gb": 6,
        "arch": "ampere",
        "bf16": True,
        "batch": 1,
        "accum": 8,
        "grad_ckpt": True,
        "backend": "hf",        # Windows нативно: transformers + peft + trl + bitsandbytes
    },
    "2x-t4": {
        "gpus": 2,
        "vram_gb": 16,
        "arch": "turing",
        "bf16": False,          # T4 не поддерживает bf16
        "batch": 2,
        "accum": 4,
        "grad_ckpt": True,
        "backend": "unsloth",   # Linux / Kaggle / Colab
    },
    "1x-t4": {
        "gpus": 1,
        "vram_gb": 16,
        "arch": "turing",
        "bf16": False,
        "batch": 2,
        "accum": 4,
        "grad_ckpt": True,
        "backend": "unsloth",
    },
}

# (размер модели в млрд параметров, минимум VRAM в ГБ)
QLORA_VRAM_GB = [(3, 3.5), (7, 5), (8, 6), (9, 6.5), (11, 7.5), (14, 8.5),
                 (27, 22), (32, 26), (40, 30), (70, 41)]
LORA16_VRAM_GB = [(3, 8), (7, 19), (8, 22), (9, 24), (11, 29), (14, 33),
                  (27, 64), (32, 76), (40, 96), (70, 164)]


def params_b(name: str):
    """Извлекает размер модели из имени: 'Qwen2.5-Coder-7B' -> 7.0."""
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", name)
    return float(m.group(1)) if m else None


def vram_need(params: float, precision: str):
    table = QLORA_VRAM_GB if precision == "4bit" else LORA16_VRAM_GB
    for size, gb in table:
        if params <= size:
            return gb
    return None
