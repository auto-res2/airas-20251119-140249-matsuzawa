"""src/train.py
Training script for OMEGA-LR and comparative baselines – fully executable, Hydra-driven.
"""
from __future__ import annotations

import math
import os
import random
import time
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from datasets import disable_caching
from omegaconf import OmegaConf
from torch.optim import AdamW

from .model import build_model_and_tokenizer, group_parameters_by_layer
from .preprocess import build_dataloaders, canonicalise_answer

# ---------------------------------------------------------------------------
# Utilities & global state
# ---------------------------------------------------------------------------
CACHE = Path(".cache").resolve()
os.environ.setdefault("HF_HOME", str(CACHE))
disable_caching()

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# OMEGA-LR controller (closed-loop, gap-aware, optimiser-aware)
# ---------------------------------------------------------------------------
class OmegaLRController:
    """Layer-wise online magnitude-equalising LR controller."""

    def __init__(self, optimizer: AdamW, param_groups: List[dict], run_cfg):
        p = run_cfg.algorithm.params
        self.opt = optimizer
        self.param_groups = param_groups
        self.rho0 = p.rho_0
        self.beta_g = p.beta_g
        self.kappa = p.kappa
        self.beta_w = p.beta_w
        self.beta_rho = p.beta_rho
        self.eps = p.eps
        self.clip_min = p.lr_clip_factor.min * run_cfg.training.optimizer.base_lr
        self.clip_max = p.lr_clip_factor.max * run_cfg.training.optimizer.base_lr
        self.w_ema = [0.0 for _ in param_groups]
        self.rho_ema = [0.0 for _ in param_groups]
        self._stats = [dict(sum=0.0, sq=0.0, n=0) for _ in param_groups]
        self.g_t = 1.0  # generalisation-gap proxy

    # -----------------------------------------------------------
    def update_gap(self, loss_tr: float, loss_val: float):
        self.g_t = 0.95 * self.g_t + 0.05 * ((loss_val + self.eps) / (loss_tr + self.eps))

    # -----------------------------------------------------------
    @torch.no_grad()
    def step(self) -> List[float]:
        beta1, beta2 = self.opt.param_groups[0]["betas"]
        rho_star = self.rho0 * (self.g_t ** (-self.beta_g))
        layer_rhos: List[float] = []

        for idx, group in enumerate(self.param_groups):
            u_sq, w_sq = 0.0, 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.opt.state[p]
                if not state:
                    continue  # state not initialised yet
                t = int(state["step"]) + 1  # avoid div/0
                m_hat = state["exp_avg"] / (1 - beta1 ** t)
                v_hat = state["exp_avg_sq"] / (1 - beta2 ** t)
                u_sq += ((m_hat / (v_hat.sqrt() + self.eps)) ** 2).sum().item()
                w_sq += (p.data ** 2).sum().item()

            u_mag = math.sqrt(u_sq) * group["lr"]
            w_mag = math.sqrt(w_sq) + self.eps

            # EMAs ------------------------------------------------
            if self.w_ema[idx] == 0.0:
                self.w_ema[idx] = w_mag
            self.w_ema[idx] = self.beta_w * self.w_ema[idx] + (1 - self.beta_w) * w_mag

            rho = u_mag / (self.w_ema[idx] + self.eps)
            if self.rho_ema[idx] == 0.0:
                self.rho_ema[idx] = rho
            self.rho_ema[idx] = self.beta_rho * self.rho_ema[idx] + (1 - self.beta_rho) * rho

            # Integral control -----------------------------------
            lr_scale = (rho_star / (rho + self.eps)) ** self.kappa
            new_lr = max(self.clip_min, min(self.clip_max, group["lr"] * lr_scale))
            group["lr"] = new_lr

            # Stats ----------------------------------------------
            st = self._stats[idx]
            st["sum"], st["sq"], st["n"] = (
                st["sum"] + rho,
                st["sq"] + rho ** 2,
                st["n"] + 1,
            )
            layer_rhos.append(rho)

        return layer_rhos

    # -----------------------------------------------------------
    def stats_summary(self) -> Tuple[float, float]:
        n = sum(s["n"] for s in self._stats)
        if n == 0:
            return 0.0, 0.0
        mean = sum(s["sum"] for s in self._stats) / n
        var = sum(s["sq"] for s in self._stats) / n - mean ** 2
        return mean, math.sqrt(max(var, 0.0))

# ---------------------------------------------------------------------------
# SIGMA-LR controller (baseline)
# ---------------------------------------------------------------------------
class SigmaLRController:
    def __init__(self, optimizer: AdamW, run_cfg):
        p = run_cfg.algorithm.params
        self.opt = optimizer
        self.threshold = p.sharpness_threshold
        self.inc = p.lr_increase_factor
        self.dec = p.lr_decrease_factor
        self.ema_beta = p.ema_beta
        self.eps = p.eps
        self.clip_min = p.lr_clip_factor.min * run_cfg.training.optimizer.base_lr
        self.clip_max = p.lr_clip_factor.max * run_cfg.training.optimizer.base_lr
        self.sharp_ema: float | None = None

    @torch.no_grad()
    def _estimate_sharpness(self, loss: torch.Tensor, grads: List[torch.Tensor]):
        grad_sq = sum((g.norm() ** 2).item() for g in grads if g is not None)
        sharp = loss.detach().item() / (math.sqrt(grad_sq) + self.eps)
        if self.sharp_ema is None:
            self.sharp_ema = sharp
        self.sharp_ema = self.ema_beta * self.sharp_ema + (1 - self.ema_beta) * sharp
        return self.sharp_ema

    @torch.no_grad()
    def step(self, loss: torch.Tensor):
        grads = [p.grad for g in self.opt.param_groups for p in g["params"]]
        s = self._estimate_sharpness(loss, grads)
        mult = self.dec if s > self.threshold else self.inc
        for g in self.opt.param_groups:
            g["lr"] = max(self.clip_min, min(self.clip_max, g["lr"] * mult))

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _greedy_generate(model, tokenizer, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor, max_new: int = 32):
    return model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=0.0,
    )

@torch.no_grad()
def _evaluate_epoch(model, tokenizer, val_loader, trial: bool = False, collect_preds: bool = False):
    device = model.device
    model.eval()
    total, correct = 0, 0
    losses: List[float] = []
    preds_table: List[Dict[str, str]] = []

    for step, batch in enumerate(val_loader):
        inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        p_lens = batch["prompt_length"].tolist()

        out = model(**inp, labels=labels)
        losses.append(out.loss.item())

        # Generation with prompt only ---------------------------------
        prompts = [seq[:pl].tolist() for seq, pl in zip(inp["input_ids"], p_lens)]
        enc = tokenizer.pad({"input_ids": prompts}, return_tensors="pt", padding=True).to(device)
        gen = _greedy_generate(model, tokenizer, enc["input_ids"], enc["attention_mask"])
        start = enc["input_ids"].shape[1]
        pred_strs = [tokenizer.decode(seq[start:], skip_special_tokens=True) for seq in gen]
        gt_strs = []
        for lab, pl in zip(labels, p_lens):
            ans_ids = lab[pl:][lab[pl:] != -100]
            gt_strs.append(tokenizer.decode(ans_ids, skip_special_tokens=True))

        for p_str, g_str in zip(pred_strs, gt_strs):
            if canonicalise_answer(p_str) == canonicalise_answer(g_str):
                correct += 1
            if collect_preds:
                preds_table.append({"prediction": p_str, "ground_truth": g_str})
        total += len(gt_strs)

        if trial and total >= 64:
            break

    em = correct / max(total, 1)
    loss = float(np.mean(losses) if losses else 0.0)
    model.train()
    return em, loss, preds_table

# ---------------------------------------------------------------------------
# Core training loop (single seed)
# ---------------------------------------------------------------------------

def _run_one_seed(run_cfg, mode: str, seed: int, wandb_run):
    _set_seed(seed)

    tokenizer, model = build_model_and_tokenizer(run_cfg)
    train_loader, val_loader = build_dataloaders(run_cfg, tokenizer)
    val_cycle = cycle(val_loader)

    param_groups = group_parameters_by_layer(model, run_cfg)
    optim = AdamW(
        param_groups,
        lr=run_cfg.training.optimizer.base_lr,
        betas=tuple(run_cfg.training.optimizer.betas),
        weight_decay=run_cfg.training.optimizer.weight_decay,
    )

    controller: Any
    if run_cfg.algorithm.name.upper().startswith("OMEGA"):
        controller = OmegaLRController(optim, param_groups, run_cfg)
    else:
        controller = SigmaLRController(optim, run_cfg)

    best_em, best_epoch = 0.0, 0
    steps_to_55: int | None = None
    step_times: List[float] = []
    torch.cuda.reset_peak_memory_stats()
    global_step = 0
    device = model.device

    for epoch in range(run_cfg.training.epochs):
        for batch_idx, batch in enumerate(train_loader):
            tic = time.perf_counter()

            inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            loss = model(**inp, labels=labels).loss / run_cfg.dataset.gradient_accumulation_steps
            loss.backward()

            if ((batch_idx + 1) % run_cfg.dataset.gradient_accumulation_steps) == 0:
                # Controller update BEFORE optimiser.step()
                if isinstance(controller, OmegaLRController):
                    val_batch = next(val_cycle)
                    v_in = {k: v.to(device) for k, v in val_batch.items() if k != "labels"}
                    v_lab = val_batch["labels"].to(device)
                    v_loss = model(**v_in, labels=v_lab).loss.item()
                    controller.update_gap(loss.item(), v_loss)
                    layer_rhos = controller.step()
                else:
                    controller.step(loss)
                    layer_rhos = []

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

                global_step += 1
                step_ms = 1e3 * (time.perf_counter() - tic)
                step_times.append(step_ms)

                if wandb_run:
                    log_dict = {
                        "train_loss": loss.item(),
                        "lr": optim.param_groups[0]["lr"],
                        "step": global_step,
                        "step_time_ms": step_ms,
                        "gpu_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
                    }
                    if layer_rhos:
                        log_dict["layer_ratio_mean"] = float(np.mean(layer_rhos))
                    wandb.log(log_dict, step=global_step)

            if mode == "trial" and global_step >= 4:
                break

        # Epoch-end validation
        val_em, val_loss, _ = _evaluate_epoch(model, tokenizer, val_loader, trial=(mode == "trial"))
        if val_em > best_em:
            best_em, best_epoch = val_em, epoch
        if steps_to_55 is None and val_em >= 0.55:
            steps_to_55 = global_step

        if wandb_run:
            wandb.log({"val_em": val_em, "val_loss": val_loss, "epoch": epoch, "step": global_step}, step=global_step)

        if mode == "trial":
            break

    # Final validation with predictions table
    val_em, val_loss, preds_tbl = _evaluate_epoch(model, tokenizer, val_loader, trial=(mode == "trial"), collect_preds=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    mean_step_ms = float(np.mean(step_times)) if step_times else 0.0
    ratio_mean, ratio_std = controller.stats_summary() if isinstance(controller, OmegaLRController) else (0.0, 0.0)

    if wandb_run:
        tbl = wandb.Table(columns=["prediction", "ground_truth", "correct"])
        correct_cnt = 0
        for row in preds_tbl:
            flag = int(canonicalise_answer(row["prediction"]) == canonicalise_answer(row["ground_truth"]))
            correct_cnt += flag
            tbl.add_data(row["prediction"], row["ground_truth"], flag)
        wandb.log({"predictions_table": tbl})
        wandb.summary.update({
            "val_em_seed": val_em,
            "best_val_em_seed": best_em,
            "steps_to_55_em_seed": steps_to_55 or -1,
            "best_epoch_seed": best_epoch,
            "gpu_peak_mem_mb_seed": peak_mem,
            "mean_step_time_ms_seed": mean_step_ms,
            "layer_update_ratio_mean_seed": ratio_mean,
            "layer_update_ratio_std_seed": ratio_std,
            "predictions_correct": correct_cnt,
            "predictions_total": len(preds_tbl),
        })

    # Memory hygiene
    del model, tokenizer, optim, controller, train_loader, val_loader
    torch.cuda.empty_cache()

    return {
        "best_em": best_em,
        "steps55": steps_to_55 or -1,
        "ratio_mean": ratio_mean,
        "gpu_mem": peak_mem,
        "t_ms": mean_step_ms,
    }

# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------

def _sample_space(trial: optuna.Trial, space: Dict[str, Dict[str, Any]]):
    out: Dict[str, Any] = {}
    for dotted, spec in space.items():
        name = dotted.replace(".", "__")
        spec_type = spec["type"].lower()
        if spec_type == "loguniform":
            val = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif spec_type == "uniform":
            val = trial.suggest_float(name, spec["low"], spec["high"], log=False)
        else:
            raise ValueError(f"Unsupported Optuna space type {spec_type}")
        out[dotted] = val
    return out


def _inject(cfg_node, params: Dict[str, Any]):
    for dotted, value in params.items():
        node = cfg_node
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value


def _objective_factory(base_cfg):
    def objective(trial: optuna.Trial):
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        sugg = _sample_space(trial, cfg.optuna.search_space)
        _inject(cfg, sugg)
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.dataset.batch_size = max(2, cfg.dataset.batch_size // 4)
        ems = [_run_one_seed(cfg, "trial", sd, None)["best_em"] for sd in cfg.training.seed_list]
        return float(np.mean(ems))
    return objective


def _run_optuna(run_cfg):
    study = optuna.create_study(direction=run_cfg.optuna.direction)
    study.optimize(_objective_factory(run_cfg), n_trials=run_cfg.optuna.n_trials, show_progress_bar=False)
    return study.best_params

# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../config", config_name="config")
def train(cfg):  # noqa: C901
    run_cfg = cfg.run  # run-specific subsection

    # Mode handling ---------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        run_cfg.optuna.n_trials = 0
        run_cfg.training.epochs = 1
        run_cfg.dataset.batch_size = max(2, run_cfg.dataset.batch_size // 4)
        run_cfg.dataset.gradient_accumulation_steps = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # Optuna ----------------------------------------------------------
    if run_cfg.optuna.n_trials and run_cfg.optuna.n_trials > 0:
        best_params = _run_optuna(run_cfg)
        _inject(run_cfg, best_params)
        run_cfg.optuna.n_trials = 0

    # WandB -----------------------------------------------------------
    wandb_run = None
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_cfg.run_id,
            mode=cfg.wandb.mode,
            resume="allow",
            config=OmegaConf.to_container(run_cfg, resolve=True),
        )
        print("WandB URL:", wandb_run.url)

    # Multi-seed execution -------------------------------------------
    seed_metrics: Dict[str, List[float]] = defaultdict(list)
    for sd in run_cfg.training.seed_list:
        res = _run_one_seed(run_cfg, cfg.mode, sd, wandb_run)
        for k, v in res.items():
            seed_metrics[k].append(v)

    if wandb_run:
        wandb.summary.update({
            "val_em_mean": float(np.mean(seed_metrics["best_em"])),
            "val_em_std": float(np.std(seed_metrics["best_em"])),
            "steps_to_55_em_mean": float(np.mean(seed_metrics["steps55"])),
            "gpu_peak_mem_mb": float(np.max(seed_metrics["gpu_mem"])),
            "mean_step_time_ms": float(np.mean(seed_metrics["t_ms"])),
            "layer_update_ratio_mean": float(np.mean(seed_metrics["ratio_mean"])),
        })
        wandb.finish()

    print(f"[Run {run_cfg.run_id} completed] EM={np.mean(seed_metrics['best_em']):.4f}")

if __name__ == "__main__":
    train()