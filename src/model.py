"""src/model.py
Model factory and parameter-group helpers.
"""
from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(run_cfg):
    tok = AutoTokenizer.from_pretrained(run_cfg.model.name, cache_dir=".cache", padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map.get(run_cfg.compute.precision.lower(), torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model.name,
        cache_dir=".cache",
        torch_dtype=dtype,
        device_map="auto",
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return tok, model

# ---------------------------------------------------------------------------
# Parameter grouping (per layer)
# ---------------------------------------------------------------------------

def _locate_blocks(model):
    for chain in [("model", "layers"), ("model", "h"), ("layers",), ("h",), ("transformer", "h")]:
        obj = model
        for attr in chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, (list, torch.nn.ModuleList)):
            return obj
    raise AttributeError("Cannot locate Transformer blocks for parameter grouping.")


def group_parameters_by_layer(model, run_cfg):
    blocks = _locate_blocks(model)
    param_groups: List[dict] = []
    seen: set[int] = set()
    for blk in blocks:
        params = list(blk.parameters(recurse=True))
        seen.update(id(p) for p in params)
        param_groups.append({"params": params, "lr": run_cfg.training.optimizer.base_lr})

    leftover = [p for p in model.parameters() if id(p) not in seen]
    if leftover:
        param_groups.append({"params": leftover, "lr": run_cfg.training.optimizer.base_lr})
    return param_groups