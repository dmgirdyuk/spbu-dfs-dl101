from __future__ import annotations

from typing import Optional

import torch
from accelerate import Accelerator
from checkpointer import CheckpointSaver
from torch import Tensor, nn, optim
from torch.nn import functional as F  # noqa
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric
from tqdm.auto import tqdm
from utils import get_logger

LOGGER = get_logger(__name__)


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    loss_fn: nn.Module,
    metric_fns: dict[str, Metric],
    lr_scheduler: ReduceLROnPlateau,
    accelerator: Accelerator,
    epoch_num: int,
    checkpointer: CheckpointSaver,
    tb_logger: Optional[SummaryWriter],  # tensorboard logger
    save_on_val: bool = True,  # saves checkpoint on the validation stage
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
            metric_fns=metric_fns,
            accelerator=accelerator,
            checkpointer=checkpointer,
            tb_logger=tb_logger,
            global_train_step=global_train_step,
            save_on_train=not save_on_val,
        )

        if val_dataloader is None:
            continue

        global_val_step, total_val_metric = validation_step(
            epoch=epoch,
            model=model,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            checkpointer=checkpointer,
            tb_logger=tb_logger,
            global_val_step=global_val_step,
            save_on_val=save_on_val,
        )

        lr_scheduler.step(total_val_metric)
        tb_logger.add_scalar(f"learning_rate", lr_scheduler.get_last_lr()[0], epoch)


def train_step(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    loss_fn: nn.Module,
    metric_fns: dict[str, Metric],
    accelerator: Accelerator,
    checkpointer: CheckpointSaver,
    tb_logger: Optional[SummaryWriter],
    global_train_step: int,
    save_on_train: bool = False,
) -> int:
    model.train()

    loss_name = loss_fn.__class__.__name__

    total_train_loss = 0.0
    total_train_metrics = {name: 0.0 for name in metric_fns}

    batches_num = 0
    for inputs, targets in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss: Tensor = loss_fn(outputs, targets)
        total_train_loss += loss.item()

        for metric_fn in metric_fns.values():
            metric_fn.update(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()

        tb_logger.add_scalar(f"{loss_name}_train_batch", loss.item(), global_train_step)

        global_train_step += 1
        batches_num += 1

    total_train_loss /= batches_num
    LOGGER.info("Train %s: %.5f", loss_name, total_train_loss)
    tb_logger.add_scalar(f"{loss_name}_train_epoch", total_train_loss, epoch)
    for name, metric_fn in metric_fns.items():
        total_train_metrics[name] = metric_fn.compute()
        metric_fn.reset()

        LOGGER.info("Train %s: %.5f", name, total_train_metrics[name])
        tb_logger.add_scalar(f"{name}_train_epoch", total_train_metrics[name], epoch)

    if save_on_train:
        checkpointer.save(
            metric_val=total_train_metrics[checkpointer.metric_name], epoch=epoch
        )

    return global_train_step


def validation_step(
    epoch: int,
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    metric_fns: dict[str, Metric],
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,  # tensorboard logger
    global_val_step: int,
    save_on_val: bool = True,  # saves checkpoint on the validation stage
) -> tuple[int, float]:
    model.eval()

    loss_name = loss_fn.__class__.__name__

    total_val_loss = 0.0
    total_val_metrics = {name: 0.0 for name in metric_fns}

    batches_num = 0
    for inputs, targets in tqdm(val_dataloader, desc="Validation"):
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_val_loss += loss.item()

            for metric_fn in metric_fns.values():
                metric_fn.update(outputs, targets)

        tb_logger.add_scalar(f"{loss_name}_val_batch", loss.item(), global_val_step)

        global_val_step += 1
        batches_num += 1

    total_val_loss /= len(val_dataloader)
    LOGGER.info("Val %s: %.5f", loss_name, total_val_loss)
    tb_logger.add_scalar(f"{loss_name}_val_epoch", total_val_loss, epoch)
    for name, metric_fn in metric_fns.items():
        total_val_metrics[name] = metric_fn.compute()
        metric_fn.reset()

        LOGGER.info("Val %s: %.5f", name, total_val_metrics[name])
        tb_logger.add_scalar(f"{name}_val_epoch", total_val_metrics[name], epoch)

    if save_on_val:
        checkpointer.save(
            metric_val=total_val_metrics[checkpointer.metric_name], epoch=epoch
        )

    return global_val_step, total_val_metrics[checkpointer.metric_name]


def count_pytorch_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
