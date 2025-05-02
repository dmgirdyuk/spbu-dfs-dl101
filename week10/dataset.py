from typing import Any

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
    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        img = Image.open(self.images[index]).convert("RGB")
        img = np.array(img, dtype=np.uint8)

        target = self.parse_voc_xml(
            element_tree_parse(self.annotations[index]).getroot()
        )
        bboxes, label_ids = self.parse_target(target=target, img=img)

        assert self.transform is not None

        augmented = self.transform(image=img, bboxes=bboxes, label_ids=label_ids)
        img = augmented["image"].float()
        target = {
            "bboxes": torch.tensor(augmented["bboxes"], dtype=torch.float32),
            "label_ids": torch.tensor(augmented["label_ids"], dtype=torch.int32),
        }

        return img, target

    @staticmethod
    def parse_target(
        target: dict[str, Any], img: ImageT
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

            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            bboxes.append([x_center, y_center, width, height])  # YOLO format

            label_ids.append(label_id)

        return bboxes, label_ids


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
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}


LABELS_ID_MAP = {
    "background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "potted plant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}
