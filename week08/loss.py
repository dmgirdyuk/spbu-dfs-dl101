from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa

EPSILON = 1e-7


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input=input, target=torch.argmax(target, dim=1))


class DiceLoss(nn.Module):
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


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 0.0,
        size_average: bool = True,
        ignore_index: int = 255,
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        targets = torch.argmax(targets, dim=1)
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="sum", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean() if self.size_average else focal_loss.sum()


class CEDiceLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        eps: float = EPSILON,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.ce = CrossEntropyLoss()
        self.dice = DiceLoss(eps=eps)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return self.alpha * self.ce(inputs, targets) + self.beta * self.dice(
            inputs, targets
        )
