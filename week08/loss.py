from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

EPSILON = 1e-7


class MulticlassCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input=input, target=torch.argmax(target, dim=1))


class MulticlassDiceLoss(nn.Module):
    def __init__(self, eps: float = EPSILON) -> None:
        super().__init__()

        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probas = F.softmax(logits, dim=1)

        intersection = (targets * probas).sum((0, 2, 3))
        cardinality = (targets + probas).sum((0, 2, 3))

        dice_coefficient = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        dice_loss = 1.0 - dice_coefficient

        mask = targets.sum((0, 2, 3)) > 0
        dice_loss = dice_loss * mask

        if mask.sum() > 0:
            return dice_loss.sum() / mask.sum()

        return dice_loss.sum()
