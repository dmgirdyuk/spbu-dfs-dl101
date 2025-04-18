from typing import cast

import torch
import torch.nn.functional as F  # noqa
from torch import Tensor, nn

EPSILON = 1e-8


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        weight: Tensor,
        ignore_index: int = 255,
        eps: float = EPSILON,
        alpha: float = 0.8,
        beta: float = 0.2,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.ce = nn.CrossEntropyLoss(
            weight=weight, reduction="mean", ignore_index=ignore_index
        )
        self.dice = DiceLoss(weight=weight, ignore_index=ignore_index, eps=eps)

    def forward(self, logits, targets):
        return self.alpha * self.ce(logits, targets) + self.beta * self.dice(
            logits, targets
        )


class DiceLoss(nn.Module):
    def __init__(
        self,
        weight: Tensor,
        ignore_index: int = 255,
        eps: float = EPSILON,
    ) -> None:
        super().__init__()

        self.weight = weight
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        logits: (N, C, H, W)
        targets: (N, H, W)
        """
        classes_num = logits.size(1)
        probs = F.softmax(logits, dim=1)

        mask = cast(Tensor, targets != self.ignore_index).unsqueeze(1)  # (N, 1, H, W)

        targets_onehot = (
            F.one_hot(
                torch.where(cast(Tensor, targets == self.ignore_index), 0, targets),
                num_classes=classes_num,
            )
            .permute(0, 3, 1, 2)
            .float()
            * mask
        )

        intersection = (probs * targets_onehot).sum(dim=(0, 2, 3))
        total = probs.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3))

        if self.weight is not None:
            intersection = intersection * self.weight
            total = total * self.weight

        dice = (2.0 * intersection + self.eps) / (total + self.eps)

        return 1 - dice.mean()
