# From https://github.com/ultralytics/ultralytics

import math
from typing import Optional, cast

import torch
import torch.nn.functional as F  # noqa
from torch import Tensor, nn
from utils import EPSILON, bbox2dist, dist2bbox, get_logger, make_anchors, xywh2xyxy
from yolo_custom import YOLO

LOGGER = get_logger(__name__)


class YOLOLoss:
    def __init__(
        self,
        model: YOLO,
        cls_coef: float = 0.5,
        box_coef: float = 7.5,
        dfl_coef: float = 1.5,
        top_k: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = EPSILON,
    ) -> None:
        self.device = next(model.parameters()).device

        self.stride: Tensor = model.head.stride
        self.classes_num: int = model.head.classes_num
        self.outputs_num: int = model.head.outputs_num
        self.reg_max = model.head.dfl.channels
        self.use_dfl = self.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            top_k=top_k, classes_num=self.classes_num, alpha=alpha, beta=beta, eps=eps
        )

        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)

        self.cls_coef = cls_coef
        self.box_coef = box_coef
        self.dfl_coef = dfl_coef

    def __call__(self, preds: list[Tensor], targets: Tensor) -> Tensor:
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        # in the ultralytics YOLO, model returns tuple of preds and features
        # feats = preds[1] if isinstance(preds, tuple) else preds
        feats = preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.outputs_num, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.classes_num), 1)  # [b, outputs_num, 8400]

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        img_size = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = self.preprocess(
            targets=targets.to(self.device),
            batch_size=batch_size,
            scale_tensor=img_size[[1, 0, 1, 0]],
        )  # xyxy, [b, gt_samples, 5]
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # xyxy, (b, num_total_anchors, 4)
        pred_bboxes = self.bbox_decode(
            anchor_points=anchor_points, pred_dist=pred_distri
        )

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.box_coef
        loss[1] *= self.cls_coef
        loss[2] *= self.dfl_coef

        return loss[0] + loss[1] + loss[2]

    def preprocess(
        self, targets: Tensor, batch_size: int, scale_tensor: Tensor
    ) -> Tensor:
        nl, ne = targets.shape  # [gt_boxes in batch, 6]
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = cast(Tensor, i == j)
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]

            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))

        return out

    def bbox_decode(self, anchor_points: Tensor, pred_dist: Tensor) -> Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )

        return dist2bbox(distance=pred_dist, anchor_points=anchor_points, xywh=False)


class BboxLoss(nn.Module):
    def __init__(self, reg_max: int = 16, eps: float = EPSILON) -> None:
        super().__init__()

        self.eps = eps

        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        target_bboxes: Tensor,
        target_scores: Tensor,
        target_scores_sum: Tensor,
        fg_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            xywh=False,
            ciou=True,
            eps=self.eps,
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


def bbox_iou(
    box1: Tensor,
    box2: Tensor,
    xywh: bool = True,
    giou: bool = False,
    diou: bool = False,
    ciou: bool = False,
    eps: float = EPSILON,
) -> Tensor:
    if xywh:  # from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if ciou or diou or giou:
        # convex (smallest enclosing box) width
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        # convex height
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

        if ciou or diou:
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if ciou:
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))

                return iou - (rho2 / c2 + v * alpha)

            return iou - rho2 / c2

        c_area = cw * ch + eps  # convex area

        return iou - (c_area - union) / c_area

    return iou


class DFLoss(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()

        self.reg_max = reg_max

    def __call__(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned
    metric, which combines both classification and localization information.

    Attributes:
        top_k (int): The number of top candidates to consider.
        classes_num (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the
            task-aligned metric.
        beta (float): The beta parameter for the localization component of the
            task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(
        self,
        top_k: int = 10,
        classes_num: int = 20,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = EPSILON,
    ) -> None:
        super().__init__()

        self.top_k = top_k
        self.classes_num = classes_num
        self.bg_idx = classes_num
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.batch_size = 0
        self.max_boxes_num = 0

    @torch.no_grad()
    def forward(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        anc_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape
                (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape
                (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape
                (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape
                (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape
                (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape
                (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape
                (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape
                (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape
                (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape
                (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape
                (bs, num_total_anchors).
        """

        self.batch_size = pd_scores.shape[0]
        self.max_boxes_num = gt_bboxes.shape[1]

        device = gt_bboxes.device

        if self.max_boxes_num == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(
                pd_scores=pd_scores,
                pd_bboxes=pd_bboxes,
                anc_points=anc_points,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                mask_gt=mask_gt,
            )
        except torch.cuda.OutOfMemoryError:
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU.")
            cpu_tensors = [
                t.cpu()
                for t in (
                    pd_scores,
                    pd_bboxes,
                    anc_points,
                    gt_labels,
                    gt_bboxes,
                    mask_gt,
                )
            ]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        anc_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores=pd_scores,
            pd_bboxes=pd_bboxes,
            anc_points=anc_points,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            mask_gt=mask_gt,
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos=mask_pos,
            overlaps=overlaps,
            n_max_boxes=self.max_boxes_num,
        )

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            target_gt_idx=target_gt_idx,
            fg_mask=fg_mask,
        )

        # normalize
        align_metric *= mask_pos
        # b, max_num_obj
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps))
            .amax(-2)
            .unsqueeze(-1)
        )
        target_scores = target_scores * norm_align_metric

        return (
            target_labels,
            target_bboxes,
            target_scores,
            fg_mask.bool(),
            target_gt_idx,
        )

    def get_pos_mask(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        anc_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Get positive mask for each ground truth box.

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape
                (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes
                with shape (bs, max_num_obj, h*w).
        """

        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        # Get top_k_metric mask, (b, max_num_obj, h*w)
        mask_top_k = self.select_top_k_candidates(
            align_metric, top_k_mask=mask_gt.expand(-1, -1, self.top_k).bool()
        )
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def select_candidates_in_gts(self, xy_centers: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape
                (b, n_boxes, 4).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """

        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat(
            (xy_centers[None] - lt, rb - xy_centers[None]), dim=2
        ).view(bs, n_boxes, n_anchors, -1)

        return bbox_deltas.amin(3).gt_(self.eps)

    def get_box_metrics(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute alignment metric given predicted and ground truth bounding boxes.

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and
                localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth
                boxes.
        """

        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros(
            [self.batch_size, self.max_boxes_num, na],
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device,
        )
        bbox_scores = torch.zeros(
            [self.batch_size, self.max_boxes_num, na],
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )

        # 2, b, max_num_obj
        ind = torch.zeros([2, self.batch_size, self.max_boxes_num], dtype=torch.long)
        # b, max_num_obj
        ind[0] = (
            torch.arange(end=self.batch_size).view(-1, 1).expand(-1, self.max_boxes_num)
        )
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls; b, max_num_obj, h*w
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.max_boxes_num, -1, -1)[
            mask_gt
        ]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    @staticmethod
    def iou_calculation(gt_bboxes: Tensor, pd_bboxes: Tensor) -> Tensor:
        """Calculate IoU for horizontal bounding boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """

        return (
            bbox_iou(gt_bboxes, pd_bboxes, xywh=False, ciou=True).squeeze(-1).clamp_(0)
        )

    def select_top_k_candidates(
        self, metrics: Tensor, largest: bool = True, top_k_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b
                is the batch size, max_num_obj is the maximum number of objects, and
                h*w represents the total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the
                smallest values.
            top_k_mask (torch.Tensor): An optional boolean tensor of shape
                (b, max_num_obj, topk), where top_k is the number of top candidates
                to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the
                selected top-k candidates.
        """

        # (b, max_num_obj, top_k)
        top_k_metrics, top_k_idxs = torch.topk(
            metrics, self.top_k, dim=-1, largest=largest
        )
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(
                top_k_idxs
            )
        # (b, max_num_obj, top_k)
        top_k_idxs.masked_fill_(~top_k_mask, 0)

        # (b, max_num_obj, top_k, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(
            metrics.shape, dtype=torch.int8, device=top_k_idxs.device
        )
        ones = torch.ones_like(
            top_k_idxs[:, :, :1], dtype=torch.int8, device=top_k_idxs.device
        )
        for k in range(self.top_k):
            # expand top_k_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, top_k_idxs[:, :, k : k + 1], ones)

        # filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    @staticmethod
    def select_highest_overlaps(
        mask_pos: Tensor, overlaps: Tensor, n_max_boxes: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape
                (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """

        # convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            # (b, n_max_boxes, h*w)
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(
                mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device
            )
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            # (b, n_max_boxes, h*w)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)

        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(
        self,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        target_gt_idx: Tensor,
        fg_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute target labels, target bounding boxes, and target scores for the positive
        anchor points.

        Returns:
            target_labels (torch.Tensor): Shape (b, h*w), containing the target labels
                for positive anchor points.
            target_bboxes (torch.Tensor): Shape (b, h*w, 4), containing the target
                bounding boxes for positive anchor points.
            target_scores (torch.Tensor): Shape (b, h*w, num_classes), containing the
                target scores for positive anchor points.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(
            end=self.batch_size, dtype=torch.int64, device=gt_labels.device
        )[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.max_boxes_num  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.classes_num),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(
            1, 1, self.classes_num
        )  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
