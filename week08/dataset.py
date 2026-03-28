import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torchvision import datasets


class CustomVOCSegmentation(datasets.VOCSegmentation):
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index]).convert("RGB")

        img = np.array(img, dtype=np.uint8)
        target = np.array(target, dtype=np.uint8)
        target = self._convert_to_segmentation_mask(target)

        assert self.transform is not None

        augmented = self.transform(image=img, mask=target)
        return augmented["image"].float(), augmented["mask"].long()

    @staticmethod
    def _convert_to_segmentation_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # Replace with class indices
        segmentation_mask = np.full(mask.shape[:2], 255, dtype=np.uint8)
        for label_idx, color in enumerate(VOC_COLORMAP):
            segmentation_mask[np.all(mask == color, axis=-1)] = label_idx

        return segmentation_mask


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


def convert_label_to_color(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    color_mask = np.full((*mask.shape[:2], 3), 0, dtype=np.uint8)
    for label_idx, color in enumerate(VOC_COLORMAP):
        color_mask[mask == label_idx] = color

    return color_mask


def compute_class_weights(
    train_dataset: CustomVOCSegmentation, classes_num: int = 21
) -> list[float]:
    # import torch
    # from tqdm import tqdm
    # class_counts = torch.zeros(classes_num)
    # for _, mask in tqdm(train_dataset):
    #     unique, counts = torch.unique(mask, return_counts=True)
    #     for cls_idx, count in zip(unique, counts):
    #         if cls_idx < classes_num:
    #             class_counts[cls_idx] += count
    #
    # weights = 1.0 / torch.log(1.02 + class_counts / class_counts.sum())
    # return (weights / weights.sum()).tolist()

    return CLASS_WEIGHTS


CLASS_WEIGHTS = [
    0.00274782744236290,
    0.05705965310335159,
    0.06798853725194931,
    0.05459493026137352,
    0.059721440076828,
    0.059969764202833176,
    0.04155288264155388,
    0.04566816985607147,
    0.033576808869838715,
    0.04967980831861496,
    0.054987259209156036,
    0.04740002378821373,
    0.042492035776376724,
    0.05335487797856331,
    0.049849510192871094,
    0.023367108777165413,
    0.058975230902433395,
    0.054136715829372406,
    0.04544247314333916,
    0.04360505938529968,
    0.053829897195100784,
]
