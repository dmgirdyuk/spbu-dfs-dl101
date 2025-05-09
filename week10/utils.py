import logging
import os
import random
from typing import Literal, cast

import cv2
import numpy as np
import torch
import torchvision
from numpy.typing import NDArray
from torch import Tensor

ImageT = NDArray[np.int8]

EPSILON = 1e-8


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Creates a logger."""
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger


def seed_everything(seed: int = 314159, torch_deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(torch_deterministic)


def make_anchors(
    x: list[Tensor], strides: Tensor, offset: float = 0.5
) -> tuple[Tensor, Tensor]:
    assert x is not None

    anchor_points, strides_ = [], []
    for i, stride in enumerate(strides):
        stride = cast(Tensor, stride)

        h, w = x[i].shape[2:4]
        dtype, device = x[0].dtype, x[0].device
        sx = torch.arange(end=w, dtype=dtype, device=device) + offset
        sy = torch.arange(end=h, dtype=dtype, device=device) + offset
        sy, sx = torch.meshgrid(sy, sx)

        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))

        strides_.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(strides_)


def non_max_suppression(
    preds: Tensor,
    conf_thd: float = 0.001,
    iou_thd: float = 0.65,
    max_wh: int = 7680,  # maximum box width and height
    max_det: int = 300,  # maximum number of boxes to keep after NMS
    max_nms: int = 30000,  # maximum number of boxes into torchvision.ops.nms()
) -> list[Tensor]:
    classes_num = preds.shape[1] - 4  # number of classes
    candidates = preds[:, 4 : 4 + classes_num].amax(1) > conf_thd  # candidates

    outputs = [torch.zeros((0, 6), device=preds.device)] * preds.shape[0]
    for idx, x in enumerate(preds):
        x = cast(Tensor, x).transpose(0, -1)[candidates[idx]]

        if not x.shape[0]:
            continue

        # detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, classes_num), 1)

        box = xywh2xyxy(box)

        if classes_num > 1:
            i, j = (cls > conf_thd).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thd]

        if not x.shape[0]:
            continue

        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou_thd)
        i = i[:max_det]
        outputs[idx] = x[i]

    return outputs


def xywh2xyxy(x: Tensor) -> Tensor:
    x_new = torch.empty_like(x)
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    x_new[..., :2] = xy - wh  # top left xy
    x_new[..., 2:] = xy + wh  # bottom right xy
    return x_new


def dist2bbox(
    distance: Tensor, anchor_points: Tensor, xywh: bool = True, dim: int = -1
) -> Tensor:
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox

    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: Tensor, bbox: Tensor, reg_max: float) -> Tensor:
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat(
        (anchor_points - x1y1, x2y2 - anchor_points), -1
    ).clamp_(0, reg_max - 0.01)


def add_bboxes_on_img(
    img: ImageT,
    bbox: list[int | float] | Tensor,
    bbox_format: Literal["xyxy", "xcycwh"],
    denormalize_bbox: bool,
    label: str,
    box_color: tuple[int, int, int] = (255, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> ImageT:
    assert len(bbox) == 4

    if bbox_format == "xcycwh":
        x_c, y_c, w, h = bbox
        x_min = x_c - w / 2
        y_min = y_c - h / 2
        x_max = x_c + w / 2
        y_max = y_c + h / 2
    elif bbox_format == "xyxy":
        x_min, y_min, x_max, y_max = bbox
    else:
        raise NotImplementedError

    if denormalize_bbox:
        img_h, img_w = img.shape[:2]
    else:
        img_h = img_w = 1

    x_min = int(x_min * img_w)
    y_min = int(y_min * img_h)
    x_max = int(x_max * img_w)
    y_max = int(y_max * img_h)

    cv2.rectangle(
        img, (x_min, y_min), (x_max, y_max), color=box_color, thickness=thickness
    )

    (text_width, text_height), _ = cv2.getTextSize(
        text=label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2
    )
    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        box_color,
        -1,
    )
    cv2.putText(
        img,
        text=label,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=text_color,
        lineType=cv2.LINE_AA,
    )
    return img
