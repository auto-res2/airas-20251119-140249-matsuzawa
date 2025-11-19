"""src/preprocess.py
GSM8K dataset preprocessing with strict leak-prevention.
"""
from __future__ import annotations

import functools
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

# ---------------------------------------------------------------------------
# Numeric canonicaliser (shared with evaluation)
# ---------------------------------------------------------------------------

def canonicalise_answer(text: str) -> str:
    import re
    from fractions import Fraction

    t = text.strip().replace(",", "")
    nums = re.findall(r"[-+]?[0-9]+\/?[0-9]*", t)
    if not nums:
        return t.lower()
    num = nums[-1]
    if "/" in num:
        n, d = map(int, num.split("/", 1))
        frac = Fraction(n, d)
        return f"{frac.numerator}/{frac.denominator}"
    return str(int(num))

# ---------------------------------------------------------------------------
# Tokenisation of a single GSM8K example
# ---------------------------------------------------------------------------

def _tokenise_gsm8k(example, tok: PreTrainedTokenizer, max_len: int):
    question = example["question"].strip()
    answer = example["answer"].split("####")[-1].strip()

    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(" " + answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    placeholder_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    placeholder_ids = [placeholder_id] * len(answer_ids)

    input_ids = prompt_ids + placeholder_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_length": len(prompt_ids),
    }

# ---------------------------------------------------------------------------
# Dataset wrapper & collator
# ---------------------------------------------------------------------------
class _Wrap(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):  # type: ignore[override]
        return self.ds[idx]


class GSMCollator:
    def __init__(self, tok: PreTrainedTokenizer):
        self.tok = tok
    def __call__(self, batch):
        ids = [b["input_ids"] for b in batch]
        lbl = [b["labels"] for b in batch]
        pl = [b["prompt_length"] for b in batch]
        enc = self.tok.pad({"input_ids": ids}, return_tensors="pt", padding=True)
        lbl_pad = self.tok.pad({"input_ids": lbl}, return_tensors="pt", padding=True)["input_ids"]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": lbl_pad,
            "prompt_length": torch.tensor(pl, dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, tokenizer: PreTrainedTokenizer) -> Tuple[DataLoader, DataLoader]:
    assert cfg.dataset.name.lower() == "gsm8k", "Only GSM8K supported."

    raw = load_dataset("openai/gsm8k", cfg.dataset.config, cache_dir=".cache/")
    train_raw = raw[cfg.dataset.train_split]
    val_raw = raw[cfg.dataset.val_split]

    map_fn = functools.partial(_tokenise_gsm8k, tok=tokenizer, max_len=cfg.dataset.max_seq_length)
    train_ds = train_raw.map(map_fn, remove_columns=train_raw.column_names)
    val_ds = val_raw.map(map_fn, remove_columns=val_raw.column_names)

    if cfg.mode == "trial":
        train_ds = train_ds.select(range(min(16, len(train_ds))))
        val_ds = val_ds.select(range(min(32, len(val_ds))))

    collator = GSMCollator(tokenizer)
    train_loader = DataLoader(_Wrap(train_ds), batch_size=cfg.dataset.batch_size, shuffle=True, pin_memory=True, collate_fn=collator)
    val_loader = DataLoader(_Wrap(val_ds), batch_size=cfg.dataset.batch_size, shuffle=False, pin_memory=True, collate_fn=collator)
    return train_loader, val_loader