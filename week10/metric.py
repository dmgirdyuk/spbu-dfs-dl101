import torch
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision


class CustomMeanAveragePrecision(MeanAveragePrecision):
    def update(self, outputs: list[Tensor], targets: Tensor) -> None:
        preds = []
        target = []
        for i, output in enumerate(outputs):
            preds.append(
                dict(
                    boxes=output[:, :4],
                    scores=output[:, 4],
                    labels=output[:, 5].to(torch.int),
                )
            )

            img_targets = targets[targets[:, 0] == i, 1:]
            target.append(
                dict(
                    boxes=img_targets[:, 1:],
                    labels=img_targets[:, 0].to(torch.int),
                )
            )

        super().update(preds=preds, target=target)
