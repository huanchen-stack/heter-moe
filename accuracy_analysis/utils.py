"""
Utility functions for quantization experiments.

Provides:
  - get_calibration_data: load WikiText-2 calibration samples
  - seed_everything: deterministic seeding for all RNGs
"""

import random

import numpy as np
import torch
from torch import Tensor


def get_calibration_data(tokenizer_path: str, nsamples: int, seed: int, seqlen: int) -> list[Tensor]:
    """Load WikiText-2 calibration data."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        trainloader.append(trainenc.input_ids[:, i:i + seqlen])
    return trainloader


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
