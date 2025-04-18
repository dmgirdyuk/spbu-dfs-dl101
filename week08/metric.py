from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric

EPSILON = 1e-8


class MeanIoU(Metric):
    def __init__(self, classes_num: int, ignore_index: int = 255) -> None:
        super().__init__()

        self.classes_num = classes_num
        self.ignore_index = ignore_index

        self.add_state(
            "confidence_matrix",
            default=torch.zeros(classes_num, classes_num, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, targets: Tensor) -> None:
        preds = preds.argmax(dim=1).flatten()
        targets = targets.flatten()

        mask = targets != self.ignore_index

        preds = preds[mask]
        targets = targets[mask]

        self.confidence_matrix += torch.bincount(
            self.classes_num * targets + preds, minlength=self.classes_num**2
        ).reshape(self.classes_num, self.classes_num)

    def compute(self) -> Tensor:
        tp = torch.diag(self.confidence_matrix)
        fp = self.confidence_matrix.sum(0) - tp
        fn = self.confidence_matrix.sum(1) - tp
        iou = tp / (tp + fp + fn + EPSILON)
        return iou.mean()
