import numpy as np
import torch
from torch.utils import data
from skimage.io import imread

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

        x, y = imread(str(input_ID)), imread(str(target_ID), as_gray=True)

        if self.transform is not None:
            x, y = self.transform(x, y)
        
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y
