# refactored code from https://github.com/mkisantal/backboned-unet
# TODO: just use 'https://github.com/huggingface/pytorch-image-models/'?


import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa
from torchvision import models
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet161_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    VGG16_Weights,
    VGG19_Weights,
)

FeaturesT = dict[str | None, torch.Tensor | None]


class UNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        classes_num: int = 21,
        decoder_filters: tuple[int] = (512, 256, 128, 64, 32),  # (256, 128, 64, 32, 16)
        parametric_upsampling: bool = False,
        skip_connection_names: list[str] | None = None,
        decoder_use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        # encoder/backbone
        self.backbone_name = backbone_name
        backbone_info = get_backbone(backbone_name, pretrained=pretrained)
        self.backbone, self.shortcut_features, self.bb_out_name = backbone_info

        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if skip_connection_names is not None:
            self.shortcut_features = skip_connection_names

        # decoder
        self.upsample_blocks = nn.ModuleList()
        # avoiding having more blocks than skip connections
        decoder_filters = decoder_filters[: len(self.shortcut_features)]
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, (filters_in, filters_out) in enumerate(
            zip(decoder_filters_in, decoder_filters)
        ):
            self.upsample_blocks.append(
                UpsampleBlock(
                    channels_in=filters_in,
                    channels_out=filters_out,
                    skip_in=shortcut_chs[num_blocks - i - 1],
                    parametric=parametric_upsampling,
                    use_bn=decoder_use_batchnorm,
                )
            )

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes_num, kernel_size=1)

        if freeze_encoder:
            self.freeze_encoder_parameters()

    def forward(self, *input):
        x, features = self.forward_backbone(*input)

        for skip_feature_name, upsample_block in zip(
            self.shortcut_features[::-1], self.upsample_blocks
        ):
            x = upsample_block(x, features[skip_feature_name])

        x = self.final_conv(x)
        return x

    def forward_backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, FeaturesT]:
        features: FeaturesT = {}
        if None in self.shortcut_features:
            features[None] = None

        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self) -> tuple[list[int], int]:
        x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = (
            self.backbone_name.startswith("vgg") or self.backbone_name == "unet_encoder"
        )
        channels = [] if has_fullres_features else [0]

        out_channels = 1
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break

        return channels, out_channels

    def freeze_encoder_parameters(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False


def get_backbone(name: str, pretrained=True) -> tuple[nn.Module, list[str | None], str]:
    if name == "resnet18":
        backbone = models.resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
    elif name == "resnet34":
        backbone = models.resnet34(
            weights=ResNet34_Weights.DEFAULT if pretrained else None
        )
    elif name == "resnet50":
        backbone = models.resnet50(
            weights=ResNet50_Weights.DEFAULT if pretrained else None
        )
    elif name == "resnet101":
        backbone = models.resnet101(
            weights=ResNet101_Weights.DEFAULT if pretrained else None
        )
    elif name == "resnet152":
        backbone = models.resnet152(
            weights=ResNet152_Weights.DEFAULT if pretrained else None
        )
    elif name == "vgg16":
        backbone = models.vgg16_bn(
            weights=VGG16_Weights.DEFAULT if pretrained else None
        ).features
    elif name == "vgg19":
        backbone = models.vgg19_bn(
            weights=VGG19_Weights.DEFAULT if pretrained else None
        ).features
    elif name == "densenet121":
        backbone = models.densenet121(
            weights=DenseNet121_Weights.DEFAULT if pretrained else None
        ).features
    elif name == "densenet161":
        backbone = models.densenet161(
            weights=DenseNet161_Weights.DEFAULT if pretrained else None
        ).features
    elif name == "densenet169":
        backbone = models.densenet169(
            weights=DenseNet169_Weights.DEFAULT if pretrained else None
        ).features
    elif name == "densenet201":
        backbone = models.densenet201(
            weights=DenseNet201_Weights.DEFAULT if pretrained else None
        ).features
    else:
        raise ValueError("Unknown backbone.")

    if name.startswith("resnet"):
        feature_names = [None, "relu", "layer1", "layer2", "layer3"]
        backbone_output = "layer4"
    elif name == "vgg16":
        feature_names = ["5", "12", "22", "32", "42"]
        backbone_output = "43"
    elif name == "vgg19":
        feature_names = ["5", "12", "25", "38", "51"]
        backbone_output = "52"
    elif name.startswith("densenet"):
        feature_names = [None, "relu0", "denseblock1", "denseblock2", "denseblock3"]
        backbone_output = "denseblock4"
    else:
        raise ValueError("Unknown backbone.")

    return backbone, feature_names, backbone_output


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int | None = None,
        skip_in: int = 0,
        use_bn: bool = True,
        parametric: bool = True,
    ) -> None:
        super().__init__()

        self.parametric = parametric
        channels_out = channels_in / 2 if channels_out is None else channels_out

        if parametric:
            # options: kernel=2 padding=0, kernel=4 padding=1
            self.up: nn.Upsample | nn.ConvTranspose2d = nn.ConvTranspose2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=not use_bn,
            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            channels_in = channels_in + skip_in
            self.conv1 = nn.Conv2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=(not use_bn),
            )

        self.bn1 = nn.BatchNorm2d(channels_out) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        conv2_in = channels_out if not parametric else channels_out + skip_in
        self.conv2 = nn.Conv2d(
            in_channels=conv2_in,
            out_channels=channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_bn,
        )
        self.bn2 = nn.BatchNorm2d(channels_out) if use_bn else nn.Identity()

    def forward(
        self, x: torch.Tensor, skip_connection: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.up(x)
        if self.parametric:
            x = self.bn1(x)
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


def count_model_params(model: nn.Module) -> int:
    """Returns the amount of pytorch model parameters."""
    return sum(p.numel() for p in model.parameters())
