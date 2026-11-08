"""Loops de treino e avaliação."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, amp: bool,
                    limit_batches: int | None = None) -> dict:
    model.train()
    tot_loss, correct, n = 0.0, 0, 0
    for b, (x, y) in enumerate(loader):
        if limit_batches is not None and b >= limit_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        tot_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return {"loss": tot_loss / max(n, 1), "acc": correct / max(n, 1)}


@torch.no_grad()
def predict(model, loader, device, amp: bool = False, limit_batches: int | None = None):
    """Retorna (logits [N,C] float32 numpy, labels [N] numpy)."""
    model.eval()
    all_logits, all_y = [], []
    for b, (x, y) in enumerate(loader):
        if limit_batches is not None and b >= limit_batches:
            break
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        all_logits.append(logits.float().cpu())
        all_y.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_y).numpy()


def eval_loss(logits: np.ndarray, y: np.ndarray) -> float:
    return float(nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y)).item())
