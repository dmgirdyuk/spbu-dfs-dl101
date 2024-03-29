from __future__ import annotations

import os
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path
from shutil import copyfile, rmtree
from typing import Any, Callable, cast

import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.nn import functional as F  # noqa
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    loss_function: Callable[[Any, Any], torch.Tensor],
    metric_function: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: LRScheduler,  # learning rate scheduler.
    accelerator: Accelerator,
    epoch_num: int,
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,  # tensorboard logger
    save_on_val: bool = True,  # saves checkpoint on the validation stage
    show_every_x_batch: int = 30,
) -> None:
    model.train()
    global_train_step, global_val_step = 0, 0
    for epoch in tqdm(range(epoch_num)):
        print("-" * 30)
        print(f"Epoch {epoch}/{epoch_num}")

        global_train_step = train_step(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            loss_function=loss_function,
            metric_function=metric_function,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            checkpointer=checkpointer,
            tb_logger=tb_logger,
            global_train_step=global_train_step,
            save_on_val=save_on_val,
            show_every_x_batch=show_every_x_batch,
        )

        if val_dataloader is None:
            continue

        global_val_step = validation_step(
            epoch=epoch,
            model=model,
            val_dataloader=val_dataloader,
            loss_function=loss_function,
            metric_function=metric_function,
            checkpointer=checkpointer,
            tb_logger=tb_logger,
            global_val_step=global_val_step,
            save_on_val=save_on_val,
        )


def train_step(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    loss_function: Callable[[Any, Any], torch.Tensor],
    metric_function: Callable[[Any, Any], torch.Tensor],
    lr_scheduler: LRScheduler,  # learning rate scheduler.
    accelerator: Accelerator,
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,  # tensorboard logger
    global_train_step: int,
    save_on_val: bool = True,  # saves checkpoint on the validation stage
    show_every_x_batch: int = 30,
) -> int:
    model.train()

    batch_idx = 0
    total_train_loss, total_train_metric = 0.0, 0.0
    for inputs, targets in tqdm(train_dataloader, desc="Training"):
        batch_idx += 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        metric = metric_function(outputs, targets)
        total_train_loss += loss.item()
        total_train_metric += metric.item()
        accelerator.backward(loss)
        optimizer.step()

        if not batch_idx % show_every_x_batch:
            print(f"Batch train loss: {loss.item():.5f}")
            print(f"Batch train metric: {metric.item():.5f}")

        if tb_logger is not None:
            tb_logger.add_scalar("loss_train_batch", loss.item(), global_train_step)
            tb_logger.add_scalar("metric_train_batch", metric.item(), global_train_step)
            global_train_step += 1

    lr_scheduler.step()
    total_train_loss /= len(train_dataloader)
    total_train_metric /= len(train_dataloader)
    print(f"Epoch train loss: {total_train_loss:.5f}")
    print(f"Epoch train metric: {total_train_metric:.5f}")
    if tb_logger is not None:
        tb_logger.add_scalar("loss_train_epoch", total_train_loss, epoch)
        tb_logger.add_scalar("metric_train_epoch", total_train_metric, epoch)

    if not save_on_val:
        checkpointer.save(metric_val=total_train_metric, epoch=epoch)

    return global_train_step


def validation_step(
    epoch: int,
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_function: Callable[[Any, Any], torch.Tensor],
    metric_function: Callable[[Any, Any], torch.Tensor],
    checkpointer: CheckpointSaver,
    tb_logger: SummaryWriter | None,  # tensorboard logger
    global_val_step: int,
    save_on_val: bool = True,  # saves checkpoint on the validation stage
) -> int:
    model.eval()

    total_val_loss, total_val_metric = 0.0, 0.0
    for inputs, targets in tqdm(val_dataloader, desc="Validation"):
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            metric = metric_function(outputs, targets)
            total_val_loss += loss.item()
            total_val_metric += metric.item()

        if tb_logger is not None:
            tb_logger.add_scalar("loss_val_batch", loss.item(), global_val_step)
            tb_logger.add_scalar("metric_val_batch", metric.item(), global_val_step)
            global_val_step += 1

    total_val_loss /= len(val_dataloader)
    total_val_metric /= len(val_dataloader)
    print(f"Epoch validation loss: {total_val_loss:.5f}")
    print(f"Epoch validation metric: {total_val_metric:.5f}")
    if tb_logger is not None:
        tb_logger.add_scalar("loss_val_epoch", total_val_loss, epoch)
        tb_logger.add_scalar("metric_val_epoch", total_val_metric, epoch)

    if save_on_val:
        checkpointer.save(metric_val=total_val_metric, epoch=epoch)

    return global_val_step


class MulticlassCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=input, target=torch.argmax(target, dim=1))


EPSILON = 1e-7


class MulticlassDiceLoss(nn.Module):
    def __init__(self, eps: float = EPSILON) -> None:
        super().__init__()

        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probas = F.softmax(logits, dim=1)

        intersection = (targets * probas).sum((0, 2, 3)).clamp_min(self.eps)
        cardinality = (targets + probas).sum((0, 2, 3)).clamp_min(self.eps)

        dice_coefficient = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        dice_loss = 1.0 - dice_coefficient

        mask = targets.sum((0, 2, 3)) > 0
        dice_loss *= mask

        return dice_loss.mean()


LongTensorT = torch.LongTensor


class IoUMetric(nn.Module):
    def __init__(
        self,
        classes_num: int,
        ignore_index: int | None = None,
        reduction: str | None = None,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()

        self.cls_num = classes_num
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.class_weights = class_weights

    @torch.no_grad()
    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # from
        # https://github.com/qubvel/segmentation_models.pytorch/blob/master
        # /segmentation_models_pytorch/metrics/functional.py

        outputs = torch.argmax(output, dim=1).long()
        targets = torch.argmax(target, dim=1).long()

        batch_size, height, width = outputs.shape

        if self.ignore_index is not None:
            ignore = cast(torch.Tensor, targets == self.ignore_index)
            outputs = torch.where(ignore, -1, outputs)
            targets = torch.where(ignore, -1, targets)

        tp_count = cast(
            LongTensorT, torch.zeros(batch_size, self.cls_num, dtype=torch.long)
        )
        fp_count = cast(
            LongTensorT, torch.zeros(batch_size, self.cls_num, dtype=torch.long)
        )
        fn_count = cast(
            LongTensorT, torch.zeros(batch_size, self.cls_num, dtype=torch.long)
        )

        for i in range(batch_size):
            target_i = targets[i]
            output_i = outputs[i]
            mask = output_i == target_i
            matched = torch.where(mask, target_i, -1)
            tp = torch.histc(
                matched.float(), bins=self.cls_num, min=0, max=self.cls_num - 1
            )
            fp = (
                torch.histc(
                    output_i.float(), bins=self.cls_num, min=0, max=self.cls_num - 1
                )
                - tp
            )
            fn = (
                torch.histc(
                    target_i.float(), bins=self.cls_num, min=0, max=self.cls_num - 1
                )
                - tp
            )

            tp_count[i] = tp.long()
            fp_count[i] = fp.long()
            fn_count[i] = fn.long()

        return _compute_iou_metric(
            tp=tp_count,
            fp=fp_count,
            fn=fn_count,
            reduction=self.reduction,
            class_weights=self.class_weights,
        )


def _compute_iou_metric(
    tp: LongTensorT,
    fp: LongTensorT,
    fn: LongTensorT,
    reduction: str | None = None,
    class_weights: list[float] | None = None,
) -> torch.Tensor:
    if class_weights is None and reduction is not None and "weighted" in reduction:
        raise ValueError(
            f"Class weights should be provided for `{reduction}` reduction."
        )

    class_weights = class_weights if class_weights is not None else 1.0
    class_weights = torch.tensor(class_weights).to(tp.device)
    class_weights = class_weights / class_weights.sum()

    if reduction == "micro":
        tp = cast(LongTensorT, tp.sum())
        fp = cast(LongTensorT, fp.sum())
        fn = cast(LongTensorT, fn.sum())
        score = _iou_score(tp, fp, fn)

    elif reduction == "macro":
        tp = cast(LongTensorT, tp.sum(0))
        fp = cast(LongTensorT, fp.sum(0))
        fn = cast(LongTensorT, fn.sum(0))
        score = _handle_zero_division(_iou_score(tp, fp, fn))
        score = (score * class_weights).mean()

    elif reduction == "weighted":
        tp = cast(LongTensorT, tp.sum(0))
        fp = cast(LongTensorT, fp.sum(0))
        fn = cast(LongTensorT, fn.sum(0))
        score = _handle_zero_division(_iou_score(tp, fp, fn))
        score = (score * class_weights).sum()

    elif reduction == "micro-imagewise":
        tp = cast(LongTensorT, tp.sum(1))
        fp = cast(LongTensorT, fp.sum(1))
        fn = cast(LongTensorT, fn.sum(1))
        score = _handle_zero_division(_iou_score(tp, fp, fn))
        score = score.mean()

    elif reduction == "macro-imagewise" or reduction == "weighted-imagewise":
        score = _iou_score(tp, fp, fn)
        score = (score.mean(0) * class_weights).mean()

    elif reduction == "none" or reduction is None:
        score = _iou_score(tp, fp, fn)

    else:
        raise ValueError(
            "`reduction` should be in [micro, macro, weighted, micro-imagewise, "
            "macro-imagesize, weighted-imagewise, none, None]"
        )

    return score


def _iou_score(tp: LongTensorT, fp: LongTensorT, fn: LongTensorT) -> torch.Tensor:
    return tp / (tp + fp + fn)


def _handle_zero_division(x: torch.Tensor) -> torch.Tensor:
    nans = torch.isnan(x)
    value = torch.tensor(0.0, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x


@dataclass
class Checkpoint:
    metric_val: float
    epoch: int
    save_path: Path


class CheckpointSaver:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        metric_name: str,
        save_dir: str,
        rm_save_dir: bool = False,
        max_history: int = 1,
        should_minimize: bool = True,
    ) -> None:
        """
        Args:
            accelerator: huggingface's accelerator
            model: model
            metric_name: name of the metric to log
            save_dir: checkpoint save dir
            max_history: number of checkpoints to store
            should_minimize: if True, metric should be minimized; false, otherwise
        """
        self._accelerator = accelerator
        self._model = model
        self.metric_name = metric_name
        self.save_dir = Path(save_dir)
        self.max_history = max_history
        self.should_minimize = should_minimize

        self._storage: list[Checkpoint] = []

        if os.path.exists(save_dir) and rm_save_dir:
            rmtree(save_dir)

        os.makedirs(save_dir, exist_ok=True)

    def save(self, metric_val: float, epoch: int) -> None:
        """Saves the checkpoint.

        Args:
            metric_val: value of the metric.
            epoch: epoch step.
        """
        save_name_prefix = f"model_e{epoch:03d}_checkpoint"
        save_path = self._save_checkpoint(
            model=self._model, epoch=epoch, save_name_prefix=save_name_prefix
        )
        self._storage.append(
            Checkpoint(metric_val=metric_val, epoch=epoch, save_path=save_path)
        )
        self._storage = sorted(
            self._storage, key=lambda x: x.metric_val, reverse=not self.should_minimize
        )
        if len(self._storage) > self.max_history:
            worst_item = self._storage.pop()
            os.remove(worst_item.save_path)

        copyfile(
            src=self._storage[0].save_path,
            dst=self.save_dir / "model_checkpoint_best.pt",
        )
        print(
            f"Best epoch {self.metric_name} value is {self._storage[0].metric_val:.4f} "
            f"on {self._storage[0].epoch} epoch"
        )

    def _save_checkpoint(
        self, model: nn.Module, epoch: int, save_name_prefix: str
    ) -> Path:
        save_path = pjoin(self.save_dir, f"{save_name_prefix}.pt")
        self._accelerator.wait_for_everyone()
        unwrapped_model = self._accelerator.unwrap_model(model)
        self._accelerator.save(
            obj={"epoch": epoch, "model_state_dict": unwrapped_model.state_dict()},
            f=save_path,
        )
        return Path(save_path)


def load_checkpoint(model: nn.Module, load_path: str) -> nn.Module:
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
