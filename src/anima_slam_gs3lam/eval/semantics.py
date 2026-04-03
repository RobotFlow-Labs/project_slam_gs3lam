"""Semantic reconstruction metrics."""

from __future__ import annotations

import torch


def mean_iou_percent(prediction: torch.Tensor, target: torch.Tensor, *, ignore_index: int | None = None) -> float:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    classes = torch.unique(torch.cat([prediction.reshape(-1), target.reshape(-1)]))
    ious: list[torch.Tensor] = []
    for class_id in classes:
        if ignore_index is not None and int(class_id.item()) == ignore_index:
            continue
        pred_mask = prediction == class_id
        target_mask = target == class_id
        intersection = torch.logical_and(pred_mask, target_mask).sum()
        union = torch.logical_or(pred_mask, target_mask).sum()
        if union > 0:
            ious.append(intersection.float() / union.float())
    if not ious:
        return 0.0
    return float(torch.stack(ious).mean().item() * 100.0)
