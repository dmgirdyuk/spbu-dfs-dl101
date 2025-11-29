# refactored version from https://github.com/jahongir7174/YOLOv8-pt/

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from utils import make_anchors


def yolo_v8_n(classes_num: int = 20) -> YOLO:
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width=width, depth=depth, classes_num=classes_num)


def yolo_v8_s(classes_num: int = 20) -> YOLO:
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width=width, depth=depth, classes_num=classes_num)


def yolo_v8_m(classes_num: int = 20) -> YOLO:
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width=width, depth=depth, classes_num=classes_num)


def yolo_v8_l(classes_num: int = 20) -> YOLO:
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width=width, depth=depth, classes_num=classes_num)


def yolo_v8_x(classes_num: int = 20) -> YOLO:
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width=width, depth=depth, classes_num=classes_num)


class YOLO(nn.Module):
    def __init__(
        self, width: list[int], depth: list[int], classes_num: int = 20
    ) -> None:
        super().__init__()

        self.net = DarkNet(width=width, depth=depth)
        self.fpn = DarkFPN(width=width, depth=depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(
            classes_num=classes_num, filters=(width[3], width[4], width[5])
        )
        self.head.stride = torch.tensor(
            [256 / x.shape[-2] for x in self.forward(img_dummy)]
        )
        self.stride = self.head.stride

        self.head.initialize_biases()

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))


class DarkNet(nn.Module):
    def __init__(self, width: list[int], depth: list[int]) -> None:
        super().__init__()

        p1 = [
            ConvBlock(
                in_channels=width[0], out_channels=width[1], kernel_size=3, stride=2
            )
        ]
        p2 = [
            ConvBlock(
                in_channels=width[1], out_channels=width[2], kernel_size=3, stride=2
            ),
            CSP(in_channels=width[2], out_channels=width[2], n=depth[0]),
        ]
        p3 = [
            ConvBlock(
                in_channels=width[2], out_channels=width[3], kernel_size=3, stride=2
            ),
            CSP(in_channels=width[3], out_channels=width[3], n=depth[1]),
        ]
        p4 = [
            ConvBlock(
                in_channels=width[3], out_channels=width[4], kernel_size=3, stride=2
            ),
            CSP(in_channels=width[4], out_channels=width[4], n=depth[2]),
        ]
        p5 = [
            ConvBlock(
                in_channels=width[4], out_channels=width[5], kernel_size=3, stride=2
            ),
            CSP(in_channels=width[5], out_channels=width[5], n=depth[0]),
            SPPF(in_channels=width[5], out_channels=width[5]),
        ]

        self.p1 = nn.Sequential(*p1)
        self.p2 = nn.Sequential(*p2)
        self.p3 = nn.Sequential(*p3)
        self.p4 = nn.Sequential(*p4)
        self.p5 = nn.Sequential(*p5)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bn_eps: float = 0.001,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=get_padding(
                kernel_size=kernel_size, padding=padding, dilation=dilation
            ),
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.norm(self.conv(x)))


def get_padding(
    kernel_size: int, padding: Optional[int] = None, dilation: int = 1
) -> int:
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1

    if padding is None:
        padding = kernel_size // 2

    return padding


class CSP(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, n: int = 1, add: bool = True
    ) -> None:
        super().__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels // 2)
        self.conv2 = ConvBlock(in_channels=in_channels, out_channels=out_channels // 2)
        self.conv3 = ConvBlock(
            in_channels=(2 + n) * out_channels // 2, out_channels=out_channels
        )

        self.res_m = nn.ModuleList(
            Residual(ch=out_channels // 2, add=add) for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class Residual(nn.Module):
    def __init__(self, ch: int, add: bool = True) -> None:
        super().__init__()

        self.add_m = add
        self.res_m = nn.Sequential(
            ConvBlock(in_channels=ch, out_channels=ch, kernel_size=3),
            ConvBlock(in_channels=ch, out_channels=ch, kernel_size=3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 5) -> None:
        super().__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=in_channels // 2)
        self.conv2 = ConvBlock(in_channels=in_channels * 2, out_channels=out_channels)
        self.res_m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkFPN(nn.Module):
    def __init__(self, width: list[int], depth: list[int]) -> None:
        super().__init__()

        self.up = nn.Upsample(size=None, scale_factor=2)
        self.h1 = CSP(
            in_channels=width[4] + width[5],
            out_channels=width[4],
            n=depth[0],
            add=False,
        )
        self.h2 = CSP(
            in_channels=width[3] + width[4],
            out_channels=width[3],
            n=depth[0],
            add=False,
        )
        self.h3 = ConvBlock(
            in_channels=width[3], out_channels=width[3], kernel_size=3, stride=2
        )
        self.h4 = CSP(
            in_channels=width[3] + width[4],
            out_channels=width[4],
            n=depth[0],
            add=False,
        )
        self.h5 = ConvBlock(
            in_channels=width[4], out_channels=width[4], kernel_size=3, stride=2
        )
        self.h6 = CSP(
            in_channels=width[4] + width[5],
            out_channels=width[5],
            n=depth[0],
            add=False,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


class Head(nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, classes_num: int, filters: tuple[int, ...]) -> None:
        super().__init__()

        self.channels = 16  # DFL channels
        self.classes_num = classes_num
        self.layers_num = len(filters)
        self.outputs_num = classes_num + self.channels * 4  # per anchor
        self.stride = torch.zeros(self.layers_num)

        c1 = max(filters[0], self.classes_num)
        c2 = max((filters[0] // 4, self.channels * 4))

        self.dfl = DFL(channels=self.channels)
        self.cls = nn.ModuleList(
            nn.Sequential(
                ConvBlock(in_channels=x, out_channels=c1, kernel_size=3),
                ConvBlock(in_channels=c1, out_channels=c1, kernel_size=3),
                nn.Conv2d(in_channels=c1, out_channels=self.classes_num, kernel_size=1),
            )
            for x in filters
        )
        self.box = nn.ModuleList(
            nn.Sequential(
                ConvBlock(in_channels=x, out_channels=c2, kernel_size=3),
                ConvBlock(in_channels=c2, out_channels=c2, kernel_size=3),
                nn.Conv2d(
                    in_channels=c2, out_channels=4 * self.channels, kernel_size=1
                ),
            )
            for x in filters
        )

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.layers_num):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)

        if self.training:
            return x

        self.anchors, self.strides = (
            x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
        )

        x = torch.cat([i.view(x[0].shape[0], self.outputs_num, -1) for i in x], 2)
        box, cls = x.split((self.channels * 4, self.classes_num), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)

        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self) -> None:
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.classes_num] = math.log(
                5 / m.classes_num / (640 / s) ** 2
            )


class DFL(nn.Module):
    """Distribution Focal Loss"""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()

        self.channels = channels
        self.conv = nn.Conv2d(
            in_channels=channels, out_channels=1, kernel_size=1, bias=False
        ).requires_grad_(False)
        x = torch.arange(channels, dtype=torch.float).view(1, channels, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(x)

    def forward(self, x: Tensor) -> Tensor:
        b, c, a = x.shape
        x = x.view(b, 4, self.channels, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)
