import random
from typing import Any, Optional

try:
    from defusedxml.ElementTree import parse as element_tree_parse
except ImportError:
    from xml.etree.ElementTree import parse as element_tree_parse

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.datasets import VOCDetection

from utils import ImageT


class CustomVOCDetectionDataset(VOCDetection):
    def __init__(
        self, *args, img_size: Optional[int] = None, mosaic: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.img_size = img_size
        self.mosaic = mosaic

        assert (self.mosaic and self.img_size is not None) or not self.mosaic

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        if self.mosaic:
            img, bboxes, label_ids = self.create_mosaic(index)
            assert (
                img.shape[0] == 2 * self.img_size and img.shape[1] == 2 * self.img_size
            )
        else:
            img_pil = Image.open(self.images[index]).convert("RGB")
            img = np.array(img_pil, dtype=np.uint8)

            target = self.parse_voc_xml(
                element_tree_parse(self.annotations[index]).getroot()
            )
            bboxes, label_ids = self.parse_target(target=target, img=img, to_yolo=True)

        assert self.transform is not None

        augmented = self.transform(image=img, bboxes=bboxes, label_ids=label_ids)
        img = augmented["image"].float() / 255.0
        target = {
            "bboxes": torch.tensor(augmented["bboxes"], dtype=torch.float32),
            "label_ids": torch.tensor(augmented["label_ids"], dtype=torch.int32),
        }

        return img, target

    @staticmethod
    def parse_target(
        target: dict[str, Any], img: ImageT, to_yolo: bool = True
    ) -> tuple[list[list[float]], list[int]]:
        annotation = target["annotation"]
        objects = annotation["object"]
        bboxes: list[list[float]] = []
        label_ids: list[int] = []

        img_height, img_width, _ = img.shape

        for obj in objects:
            label_name = obj["name"]

            if label_name not in LABELS_ID_MAP:
                continue

            label_id = LABELS_ID_MAP[label_name]

            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])

            if to_yolo:
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                bboxes.append([x_center, y_center, width, height])  # YOLO format
            else:
                bboxes.append([xmin, ymin, xmax, ymax])

            label_ids.append(label_id)

        return bboxes, label_ids

    def create_mosaic(self, index: int) -> tuple[ImageT, list[list[float]], list[int]]:
        mosaic_img, mosaic_bboxes, mosaic_labels = self._create_mosaic(index=index)
        mosaic_bboxes, mosaic_labels = _postprocess_mosaic_bboxes(
            mosaic_bboxes=mosaic_bboxes,
            mosaic_labels=mosaic_labels,
            input_size=self.img_size,
        )
        return mosaic_img, mosaic_bboxes, mosaic_labels

    def _create_mosaic(self, index: int) -> tuple[ImageT, list[list[float]], list[int]]:
        img_size: int = self.img_size

        mosaic_img = np.full((2 * img_size, 2 * img_size, 3), 0, dtype=np.uint8)
        border = [-img_size // 2, -img_size // 2]

        xc = int(random.uniform(-border[0], 2 * img_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * img_size + border[1]))

        indices = [index] + random.choices(range(len(self)), k=3)
        random.shuffle(indices)

        mosaic_bboxes, mosaic_labels = [], []
        for i, index in enumerate(indices):
            img_pil = Image.open(self.images[index]).convert("RGB")
            img = np.array(img_pil, dtype=np.uint8)
            target = self.parse_voc_xml(
                element_tree_parse(self.annotations[index]).getroot()
            )
            bboxes, label_ids = self.parse_target(target=target, img=img, to_yolo=False)

            shape = img.shape
            if i == 0:  # top left
                coords_a = max(xc - shape[1], 0), max(yc - shape[0], 0), xc, yc
                coords_b = (
                    shape[1] - (coords_a[2] - coords_a[0]),
                    shape[0] - (coords_a[3] - coords_a[1]),
                    shape[1],
                    shape[0],
                )
            elif i == 1:  # top right
                coords_a = (
                    xc,
                    max(yc - shape[0], 0),
                    min(xc + shape[1], img_size * 2),
                    yc,
                )
                coords_b = (
                    0,
                    shape[0] - (coords_a[3] - coords_a[1]),
                    min(shape[1], coords_a[2] - coords_a[0]),
                    shape[0],
                )
            elif i == 2:  # bottom left
                coords_a = (
                    max(xc - shape[1], 0),
                    yc,
                    xc,
                    min(img_size * 2, yc + shape[0]),
                )
                coords_b = (
                    shape[1] - (coords_a[2] - coords_a[0]),
                    0,
                    shape[1],
                    min(coords_a[3] - coords_a[1], shape[0]),
                )
            else:  # bottom right
                coords_a = (
                    xc,
                    yc,
                    min(xc + shape[1], img_size * 2),
                    min(img_size * 2, yc + shape[0]),
                )
                coords_b = (
                    0,
                    0,
                    min(shape[1], coords_a[2] - coords_a[0]),
                    min(coords_a[3] - coords_a[1], shape[0]),
                )

            x1a, y1a, x2a, y2a = coords_a
            x1b, y1b, x2b, y2b = coords_b

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            # labels
            for bbox in bboxes:
                bbox[0] = bbox[0] + pad_w
                bbox[1] = bbox[1] + pad_h
                bbox[2] = bbox[2] + pad_w
                bbox[3] = bbox[3] + pad_h

            mosaic_bboxes.extend(bboxes)
            mosaic_labels.extend(label_ids)

        return mosaic_img, mosaic_bboxes, mosaic_labels


def _postprocess_mosaic_bboxes(
    mosaic_bboxes, mosaic_labels, input_size: int
) -> tuple[list[list[float]], list[int]]:
    mosaic_bboxes_, mosaic_labels_ = [], []
    for bbox, label in zip(mosaic_bboxes, mosaic_labels):
        bbox[0] = max(bbox[0], 0)
        bbox[1] = max(bbox[1], 0)
        bbox[2] = min(bbox[2], 2 * input_size)
        bbox[3] = min(bbox[3], 2 * input_size)

        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            continue

        x_center = (bbox[0] + bbox[2]) / 2 / (2 * input_size)
        y_center = (bbox[1] + bbox[3]) / 2 / (2 * input_size)
        width = (bbox[2] - bbox[0]) / (2 * input_size)
        height = (bbox[3] - bbox[1]) / (2 * input_size)

        bbox[0] = x_center
        bbox[1] = y_center
        bbox[2] = width
        bbox[3] = height

        mosaic_bboxes_.append(bbox)
        mosaic_labels_.append(label)

    return mosaic_bboxes_, mosaic_labels_


def detection_collate_fn(batch):
    imgs, targets = zip(*batch)

    batch_targets = []
    for batch_idx, target in enumerate(targets):
        if len(target["label_ids"]) > 0:
            # [img_idx, class, x, y, w, h] format
            img_indices = torch.full(
                (len(target["label_ids"]), 1),
                fill_value=batch_idx,
                device=target["label_ids"].device,
            )
            target_tensor = torch.cat(
                [
                    img_indices,
                    target["label_ids"].unsqueeze(1).float(),
                    target["bboxes"],
                ],
                dim=1,
            )
            batch_targets.append(target_tensor)

    return (
        torch.stack(imgs),
        torch.cat(batch_targets) if batch_targets else torch.zeros(0, 6),
    )


ID_LABELS_MAP = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "pottedplant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tvmonitor",
}


LABELS_ID_MAP = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "potted plant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}
