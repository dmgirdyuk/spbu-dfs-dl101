from __future__ import annotations

import os
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path
from shutil import rmtree, copyfile

import torch
from accelerate import Accelerator
from torch import nn


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
