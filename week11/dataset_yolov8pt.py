import math
import random
from typing import Any, Optional

try:
    from defusedxml.ElementTree import parse as element_tree_parse
except ImportError:
    from xml.etree.ElementTree import parse as element_tree_parse

import cv2
import numpy as np
import torch
from torchvision.datasets import VOCDetection


class CustomVOCDetectionDataset(VOCDetection):
    def __init__(
        self, *args, img_size: Optional[int] = None, mosaic: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.img_size = img_size
        self.mosaic = mosaic
        self.augment = mosaic

        self.albumentations = Albumentations()

        self.indices = range(len(self.images))

        assert (self.mosaic and self.img_size is not None) or not self.mosaic

    def __getitem__(self, index: int):
        index = self.indices[index]

        mosaic = self.mosaic and random.random() < 1.0

        if mosaic:
            shapes = None
            # Load MOSAIC
            image, label = self.load_mosaic(index)
        else:
            # Load image
            image, shape = self.load_image(index)
            h, w = image.shape[:2]

            # Resize
            image, ratio, pad = resize(image, self.img_size, self.augment)
            shapes = shape, (
                (h / shape[0], w / shape[1]),
                pad,
            )  # for COCO mAP rescaling

            # Label
            label_init = self.parse_voc_xml(
                element_tree_parse(self.annotations[index]).getroot()
            )
            bboxes_, label_ids = parse_target(
                target=label_init, img_height=shape[0], img_width=shape[1], to_yolo=True
            )
            if not len(bboxes_):
                label = np.zeros((0, 5), dtype=np.float32)
            else:
                label = np.zeros((len(bboxes_), 5), dtype=np.float32)
                label[:, 0] = np.array(label_ids, dtype=np.float32)
                label[:, 1:] = np.array(bboxes_, dtype=np.float32)

            if label.size:
                label[:, 1:] = wh2xy(
                    label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1]
                )
            if self.augment:
                image, label = random_perspective(image, label)

        nl = len(label)  # number of labels
        if nl:
            label[:, 1:5] = xy2wh(label[:, 1:5], image.shape[1], image.shape[0])

        if self.augment:
            # Albumentations
            image, label = self.albumentations(image, label)
            nl = len(label)  # update after albumentations
            # HSV color-space
            augment_hsv(image)
            # Flip left-right
            if random.random() < 0.5:
                image = np.fliplr(image)
                if nl:
                    label[:, 1] = 1 - label[:, 1]

        target = torch.zeros((nl, 6))
        if nl:
            target[:, 1:] = torch.from_numpy(label)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)

        return torch.from_numpy(sample).float() / 255, target  # , shapes

    def __len__(self):
        return len(self.images)

    def load_image(self, i):
        image = cv2.imread(self.images[i])
        h, w = image.shape[:2]
        r = self.img_size / max(h, w)
        if r != 1:
            image = cv2.resize(
                image,
                dsize=(int(w * r), int(h * r)),
                interpolation=resample() if self.augment else cv2.INTER_LINEAR,
            )
        return image, (h, w)

    def load_mosaic(self, index):
        label4 = []
        image4 = np.full((self.img_size * 2, self.img_size * 2, 3), 0, dtype=np.uint8)
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        border = [-self.img_size // 2, -self.img_size // 2]

        xc = int(random.uniform(-border[0], 2 * self.img_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.img_size + border[1]))

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load image
            image, orig_shape = self.load_image(index)
            shape = image.shape
            if i == 0:  # top left
                x1a = max(xc - shape[1], 0)
                y1a = max(yc - shape[0], 0)
                x2a = xc
                y2a = yc
                x1b = shape[1] - (x2a - x1a)
                y1b = shape[0] - (y2a - y1a)
                x2b = shape[1]
                y2b = shape[0]
            if i == 1:  # top right
                x1a = xc
                y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], self.img_size * 2)
                y2a = yc
                x1b = 0
                y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a)
                y2b = shape[0]
            if i == 2:  # bottom left
                x1a = max(xc - shape[1], 0)
                y1a = yc
                x2a = xc
                y2a = min(self.img_size * 2, yc + shape[0])
                x1b = shape[1] - (x2a - x1a)
                y1b = 0
                x2b = shape[1]
                y2b = min(y2a - y1a, shape[0])
            if i == 3:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + shape[1], self.img_size * 2)
                y2a = min(self.img_size * 2, yc + shape[0])
                x1b = 0
                y1b = 0
                x2b = min(shape[1], x2a - x1a)
                y2b = min(y2a - y1a, shape[0])

            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            # Labels
            label_init = self.parse_voc_xml(
                element_tree_parse(self.annotations[index]).getroot()
            )
            bboxes_, label_ids = parse_target(
                target=label_init,
                img_height=orig_shape[0],
                img_width=orig_shape[1],
                to_yolo=True,
            )
            if not len(bboxes_):
                label = np.zeros((0, 5), dtype=np.float32)
            else:
                label = np.zeros((len(bboxes_), 5), dtype=np.float32)
                label[:, 0] = np.array(label_ids, dtype=np.float32)
                label[:, 1:] = np.array(bboxes_, dtype=np.float32)

            if len(label):
                label[:, 1:] = wh2xy(label[:, 1:], shape[1], shape[0], pad_w, pad_h)
            label4.append(label)

        # Concat/clip labels
        label4 = np.concatenate(label4, 0)
        for x in label4[:, 1:]:
            np.clip(x, 0, 2 * self.img_size, out=x)

        # Augment
        image4, label4 = random_perspective(image4, label4, border)

        return image4, label4


def parse_target(
    target: dict[str, Any], img_height, img_width, to_yolo: bool = True
) -> tuple[list[list[float]], list[int]]:
    annotation = target["annotation"]
    objects = annotation["object"]
    bboxes: list[list[float]] = []
    label_ids: list[int] = []

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


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
    return y


def xy2wh(x, w=640, h=640):
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1e-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1e-3)

    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y


def resample():
    choices = (
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LINEAR,
        cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4,
    )
    return random.choice(seq=choices)


def augment_hsv(image, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4):
    # HSV color-space augmentation
    h = hsv_h
    s = hsv_s
    v = hsv_v

    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype("uint8")
    lut_s = np.clip(x * r[1], 0, 255).astype("uint8")
    lut_v = np.clip(x * r[2], 0, 255).astype("uint8")

    im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def resize(image, input_size, augment):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(
            image, dsize=pad, interpolation=resample() if augment else cv2.INTER_LINEAR
        )
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT
    )  # add border
    return image, (r, r), (w, h)


def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)
    )


def random_perspective(
    samples,
    targets,
    border=(0, 0),
    degrees=0.0,
    scale=0.5,
    shear_=0.0,
    translate_=0.1,
):
    h = samples.shape[0] + border[0] * 2
    w = samples.shape[1] + border[1] * 2

    # Center
    center = np.eye(3)
    center[0, 2] = -samples.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -samples.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = np.eye(3)

    # Rotation and Scale
    rotate = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = np.eye(3)
    shear[0, 1] = math.tan(random.uniform(-shear_, shear_) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-shear_, shear_) * math.pi / 180)

    # Translation
    translate = np.eye(3)
    translate[0, 2] = random.uniform(0.5 - translate_, 0.5 + translate_) * w
    translate[1, 2] = random.uniform(0.5 - translate_, 0.5 + translate_) * h

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (
        (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any()
    ):  # image changed
        samples = cv2.warpAffine(
            samples, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0)
        )

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

        # filter candidates
        indices = candidates(box1=targets[:, 1:5].T * s, box2=new.T)
        targets = targets[indices]
        targets[:, 1:5] = new[indices]

    return samples, targets


class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations as album

            transforms = [
                album.Blur(p=0.01),
                album.CLAHE(p=0.01),
                album.ToGray(p=0.01),
                album.MedianBlur(p=0.01),
            ]
            self.transform = album.Compose(
                transforms, album.BboxParams("yolo", ["class_labels"])
            )

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image, label):
        if self.transform:
            x = self.transform(
                image=image, bboxes=label[:, 1:], class_labels=label[:, 0]
            )
            image = x["image"]
            label = np.array([[c, *b] for c, b in zip(x["class_labels"], x["bboxes"])])
        return image, label


def detection_collate_fn(batch):
    samples, targets = zip(*batch)
    for i, item in enumerate(targets):
        item[:, 0] = i  # add target image index

    return torch.stack(samples, 0), torch.cat(targets, 0)


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
