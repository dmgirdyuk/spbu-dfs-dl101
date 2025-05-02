from __future__ import annotations

from typing import Optional

import torch
from accelerate import Accelerator
from torch import Tensor, nn, optim
from torch.nn import functional as F  # noqa
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric
from tqdm.auto import tqdm

from checkpointer import CheckpointSaver
from loss import YOLOLoss
from utils import get_logger, non_max_suppression, wh2xy

LOGGER = get_logger(__name__)


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    loss_fn: nn.Module | YOLOLoss,
    metric_fn: Metric,
    lr_scheduler: ReduceLROnPlateau,
    accelerator: Accelerator,
    epoch_num: int,
    checkpointer: CheckpointSaver,
    tb_logger: Optional[SummaryWriter],
    conf_thd: float = 0.001,
    iou_thd: float = 0.65,
) -> None:
    LOGGER.info(
        "Trainable parameters in the model: %d", count_pytorch_model_params(model)
    )

    global_train_step, global_val_step = 0, 0
    for epoch in tqdm(range(epoch_num)):
        LOGGER.info("Epoch %d/%d", epoch + 1, epoch_num)

        global_train_step = train_step(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            accelerator=accelerator,
            tb_logger=tb_logger,
            global_train_step=global_train_step,
        )

        if val_dataloader is None:
            continue

        global_val_step, total_val_metric = validation_step(
            epoch=epoch,
            model=model,
            val_dataloader=val_dataloader,
            metric_fn=metric_fn,
            checkpointer=checkpointer,
            tb_logger=tb_logger,
            global_val_step=global_val_step,
            conf_thd=conf_thd,
            iou_thd=iou_thd,
        )

        lr_scheduler.step(total_val_metric)
        tb_logger.add_scalar(f"learning_rate", lr_scheduler.get_last_lr()[0], epoch)


def train_step(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    loss_fn: nn.Module | YOLOLoss,
    accelerator: Accelerator,
    tb_logger: Optional[SummaryWriter],
    global_train_step: int,
) -> int:
    model.train()

    loss_name = loss_fn.__class__.__name__

    total_train_loss = 0.0

    batches_num = 0
    for inputs, targets in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss: Tensor = loss_fn(outputs, targets)
        total_train_loss += loss.item()

        accelerator.backward(loss)
        optimizer.step()

        tb_logger.add_scalar(f"{loss_name}_train_batch", loss.item(), global_train_step)

        global_train_step += 1
        batches_num += 1

    total_train_loss /= batches_num
    LOGGER.info("Train %s: %.5f", loss_name, total_train_loss)
    tb_logger.add_scalar(f"{loss_name}_train_epoch", total_train_loss, epoch)

    return global_train_step


def validation_step(
    epoch: int,
    model: nn.Module,
    val_dataloader: DataLoader,
    metric_fn: Metric,
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,
    global_val_step: int,
    conf_thd: float = 0.001,
    iou_thd: float = 0.65,
) -> tuple[int, float]:
    model.eval()

    batches_num = 0
    for inputs, targets in tqdm(val_dataloader, desc="Validation"):
        with torch.no_grad():
            outputs = model(inputs)

            outputs = non_max_suppression(
                preds=outputs, conf_thd=conf_thd, iou_thd=iou_thd
            )
            _, _, h, w = inputs.shape
            targets[:, 2:] = wh2xy(targets[:, 2:])
            targets[:, 2:] *= torch.tensor((w, h, w, h)).to(targets.device)

            metric_fn.update(outputs, targets)

        global_val_step += 1
        batches_num += 1

    total_val_metrics: dict[str, Tensor] = metric_fn.compute()
    metric_fn.reset()
    for name, metric_val in total_val_metrics.items():
        if metric_val.ndim != 0:
            continue

        LOGGER.info("Val %s: %.5f", name, metric_val)
        tb_logger.add_scalar(f"{name}_val_epoch", metric_val, epoch)

    main_val_metric = total_val_metrics[checkpointer.metric_name].item()
    checkpointer.save(metric_val=main_val_metric, epoch=epoch)

    return global_val_step, main_val_metric


def count_pytorch_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
