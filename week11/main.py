import os
from os.path import join as pjoin

import albumentations as A
import numpy as np
import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from checkpointer import CheckpointSaver, load_checkpoint
from dataset import ID_LABELS_MAP, CustomVOCDetectionDataset, detection_collate_fn
from loss import YOLOLoss
from metric import CustomMeanAveragePrecision
from train import train
from utils import add_bboxes_on_img, non_max_suppression, seed_everything
from yolo_custom import yolo_v8_n


CLASSES_NUM = 20

LEARNING_RATE_SGD = 1e-2
LEARNING_RATE_ADAM = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-5
MOMENTUM_SGD = 0.93
BETAS_ADAM = (0.9, 0.999)
BATCH_SIZE = 32
NUM_WORKERS = 0
GRAD_ACCUMULATION_STEPS = 1
EPOCH_NUM = 100
SCHEDULER_PATIENCE = 5
SCHEDULER_GAMMA = 0.5
CHECKPOINTS_DIR = "checkpoints"
TENSORBOARD_DIR = "tensorboard"
RM_CHECKPOINTS_DIR = False

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


if __name__ == "__main__":
    seed_everything(42, torch_deterministic=False)

    IMAGE_SIZE = 640

    # COCO bbox format: [x_min, y_min, width, height] (normalized)
    bbox_params = A.BboxParams(
        format="yolo", min_area=16, min_visibility=0.1, label_fields=["label_ids"]
    )

    train_transforms = A.Compose(
        [
            # geometric transforms
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            # spatial transforms
            A.AtLeastOneBBoxRandomCrop(
                height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0
            ),
            # color transforms
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=10,
                        p=0.5,
                    ),
                ],
                p=0.5,
            ),
            # conversion
            A.ToTensorV2(),
        ],
        bbox_params=bbox_params,
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.ToTensorV2(),
        ],
        bbox_params=bbox_params,
    )

    train_dataset = CustomVOCDetectionDataset(
        root="data",
        year="2012",
        image_set="train",
        download=False,  # True
        transform=train_transforms,  # transform, not transforms!
        mosaic=True,
        img_size=IMAGE_SIZE,
    )

    val_dataset = CustomVOCDetectionDataset(
        root="data",
        year="2012",
        image_set="val",
        download=False,
        transform=val_transforms,
        mosaic=False,
        img_size=IMAGE_SIZE,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    model = yolo_v8_n(classes_num=20)
    model = load_checkpoint(
        model=model,
        load_path=pjoin(CHECKPOINTS_DIR, "model_checkpoint_best.pt"),
    )

    metric_fn = CustomMeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE_SGD,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM_SGD,
        nesterov=True,
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=SCHEDULER_GAMMA,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )
    accelerator = Accelerator(
        cpu="cpu" == DEVICE,
        mixed_precision="no",
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    )
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    )

    loss_fn = YOLOLoss(model=model)  # after accelerate!

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    checkpointer = CheckpointSaver(
        accelerator=accelerator,
        model=model,
        metric_name="map",
        save_dir=CHECKPOINTS_DIR,
        rm_save_dir=RM_CHECKPOINTS_DIR,
        max_history=5,
        should_minimize=False,
    )

    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    tensorboard_logger = torch.utils.tensorboard.SummaryWriter(log_dir=TENSORBOARD_DIR)

    train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        epoch_num=EPOCH_NUM,
        checkpointer=checkpointer,
        tb_logger=tensorboard_logger,
    )
