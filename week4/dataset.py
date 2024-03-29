import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torchvision import datasets

LABELS_MAP = {
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
    20: "tv/monitor",
}


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class CustomVOCSegmentation(datasets.VOCSegmentation):
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index]).convert("RGB")

        img = np.array(img, dtype=np.uint8)
        target = np.array(target, dtype=np.uint8)
        target = self._convert_to_segmentation_mask(target)

        assert self.transform is not None

        augmented = self.transform(image=img, mask=target)
        return augmented["image"].float(), augmented["mask"].float().permute(2, 0, 1)

    @staticmethod
    def _convert_to_segmentation_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.uint8)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(np.uint8)

        return segmentation_mask
