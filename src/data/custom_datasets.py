import numpy as np
import torch
from torch.utils import data
from skimage.io import imread
from PIL import Image

class SegmentationDataSet(data.Dataset):
    def __init__(self, inputs: list, targets: list, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # x, y = imread(str(input_ID)), imread(str(target_ID), as_gray=True)
        image = np.array(Image.open(str(input_ID)).convert("RGB"))
        mask = np.array(Image.open(str(target_ID)).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask

