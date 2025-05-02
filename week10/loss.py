import math

import torch
from torch import Tensor
from torch._C._nn import one_hot
from torch.nn.functional import cross_entropy

from utils import make_anchors, wh2xy
from yolo_custom import YOLO

EPSILON = 1e-8


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
        super().__init__()

        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.stride: Tensor = model.head.stride
        self.classes_num: int = model.head.classes_num
        self.outputs_num: int = model.head.outputs_num

        self.device = next(model.parameters()).device
        self.bs = 1
        self.num_max_boxes = 0

        # task aligned assigner
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # DFL params
        self.dfl_ch = model.head.dfl.channels
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=self.device)

        # losses coefs
        self.cls_coef = cls_coef
        self.box_coef = box_coef
        self.dfl_coef = dfl_coef

    def __call__(self, outputs: tuple | Tensor, targets: Tensor) -> Tensor:
        x = outputs[1] if isinstance(outputs, tuple) else outputs
        output = torch.cat([i.view(x[0].shape[0], self.outputs_num, -1) for i in x], 2)
        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.classes_num), 1)

        pred_output = pred_output.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        size = size * self.stride[0]

        anchor_points, stride_tensor = make_anchors(
            x=x, strides=self.stride, offset=0.5
        )

        # targets
        if targets.shape[0] == 0:
            gt = torch.zeros(pred_scores.shape[0], 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            gt = torch.zeros(pred_scores.shape[0], counts.max(), 5, device=self.device)
            for j in range(pred_scores.shape[0]):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]

            gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))

        gt_labels, gt_bboxes = gt.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # boxes
        b, a, c = pred_output.shape
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))

        a, b = torch.split(pred_bboxes, 2, -1)
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b), -1)

        scores = pred_scores.detach().sigmoid()
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        target_bboxes, target_scores, fg_mask = self.assign(
            scores, bboxes, gt_labels, gt_bboxes, mask_gt, anchor_points * stride_tensor
        )

        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()

        # cls loss
        loss_cls = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        loss_cls = loss_cls.sum() / target_scores_sum

        # box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            # IoU loss
            weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_box = self.iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_box = ((1.0 - loss_box) * weight).sum() / target_scores_sum

            # DFL loss
            a, b = torch.split(target_bboxes, 2, -1)
            target_lt_rb = torch.cat((anchor_points - a, b - anchor_points), -1)
            # distance (left_top, right_bottom)
            target_lt_rb = target_lt_rb.clamp(0, self.dfl_ch - 1.01)
            loss_dfl = self.dfl(
                pred_output[fg_mask].view(-1, self.dfl_ch), target_lt_rb[fg_mask]
            )
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        loss_cls *= self.cls_coef
        loss_box *= self.box_coef
        loss_dfl *= self.dfl_coef

        return loss_cls + loss_box + loss_dfl

    @torch.no_grad()
    def assign(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        true_labels: Tensor,
        true_bboxes: Tensor,
        true_mask: Tensor,
        anchors: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Task-aligned one-stage object detection assigner."""
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)

        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (
                torch.full_like(pred_scores[..., 0], self.classes_num).to(device),
                torch.zeros_like(pred_scores).to(device),
                torch.zeros_like(pred_scores[..., 0]).to(device),
            )

        i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        i[1] = true_labels.long().squeeze(-1)

        overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        overlaps = overlaps.squeeze(3).clamp(0)
        align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(
            self.beta
        )
        bs, n_boxes, _ = true_bboxes.shape
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        mask_in_gts = (
            bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        )
        metrics = align_metric * mask_in_gts
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool()
        num_anchors = metrics.shape[-1]
        top_k_metrics, top_k_indices = torch.topk(
            metrics, self.top_k, dim=-1, largest=True
        )
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile(
                [1, 1, self.top_k]
            )

        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)

        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)

        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * true_mask

        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat(
                [1, self.num_max_boxes, 1]
            )
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)

        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # assigned target labels, (b, 1)
        batch_index = torch.arange(
            end=self.bs, dtype=torch.int64, device=true_labels.device
        )[..., None]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes
        target_labels = true_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = one_hot(target_labels, self.classes_num)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.classes_num)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(-2)
        norm_align_metric = norm_align_metric.unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()

    @staticmethod
    def dfl(pred_dist: Tensor, target: Tensor) -> Tensor:
        """Return sum of left and right DFL losses.

        Distribution Focal Loss: https://ieeexplore.ieee.org/document/9792391
        """

        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        l_loss = cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        r_loss = cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (l_loss * wl + r_loss * wr).mean(-1, keepdim=True)

    @staticmethod
    def iou(box1: Tensor, box2: Tensor, eps: float = 1e-7) -> Tensor:
        """Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)."""

        # get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # intersection area
        area1 = b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)
        area2 = b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        intersection = area1.clamp(0) * area2.clamp(0)

        # union area
        union = w1 * h1 + w2 * h2 - intersection + eps

        # IoU
        iou = intersection / union
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height

        # (optional) complete IoU: https://arxiv.org/abs/1911.08287v1
        c2 = cw**2 + ch**2 + eps  # convex diagonal squared

        # center dist ** 2
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
        ) / 4
        v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))

        return iou - (rho2 / c2 + v * alpha)
