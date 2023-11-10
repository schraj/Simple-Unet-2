import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from skimage.transform import resize
import src.data.constants as c
from src.data.data_utils import get_file_list
from src.data.custom_datasets import SegmentationDataSet
from src.data.transformations import (
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
)

train_size = 0.8
random_seed = 42

class DatasetPreparer():
    def __init__(self):
        pass
    def prepare_training_dataloaders(self):
    
      self.transforms_training = ComposeDouble(
            [
                FunctionWrapperDouble(
                    resize, input=True, target=False, output_shape=(512, 512, 3)
                ),
                FunctionWrapperDouble(
                    resize,
                    input=False,
                    target=True,
                    output_shape=(512, 512),
                    order=0,
                    anti_aliasing=False,
                    preserve_range=True,
                ),
                FunctionWrapperDouble(
                    np.moveaxis, input=True, target=False, source=-1, destination=0
                ),
                FunctionWrapperDouble(normalize_01),
            ]
        )

      self.transforms_validation = ComposeDouble(
            [
                FunctionWrapperDouble(
                    resize, input=True, target=False, output_shape=(512, 512, 3)
                ),
                FunctionWrapperDouble(
                    resize,
                    input=False,
                    target=True,
                    output_shape=(512, 512),
                    order=0,
                    anti_aliasing=False,
                    preserve_range=True,
                ),
                FunctionWrapperDouble(
                    np.moveaxis, input=True, target=False, source=-1, destination=0
                ),
                FunctionWrapperDouble(normalize_01),
            ]
        )

      self.inputs = get_file_list(c.SEGMENTATION_IMAGE_DIR)
      self.inputs_train, self.inputs_valid = train_test_split(
          self.inputs, random_state=random_seed, train_size=train_size, shuffle=True
      )

      self.targets = get_file_list(c.SEGMENTATION_MASK_DIR)
      self.targets_train, self.targets_valid = train_test_split(
          self.targets, random_state=random_seed, train_size=train_size, shuffle=True
      )

      self.dataset_train = SegmentationDataSet(
          inputs=self.inputs_train, targets=self.targets_train, transform=self.transforms_training
      )

      self.dataset_valid = SegmentationDataSet(
          inputs=self.inputs_valid, targets=self.targets_valid, transform=self.transforms_validation
      )

      self.dataloader_training = DataLoader(dataset=self.dataset_train, batch_size=2, shuffle=True)

      self.dataloader_validation = DataLoader(dataset=self.dataset_valid, batch_size=2, shuffle=True)

    def prepare_test_dataloader(self, inputs, targets):
        transforms_test = ComposeDouble(
            [
                FunctionWrapperDouble(
                    resize, input=True, target=False, output_shape=(512, 512, 3)
                ),
                FunctionWrapperDouble(
                    resize,
                    input=False,
                    target=True,
                    output_shape=(512, 512),
                    order=0,
                    anti_aliasing=False,
                    preserve_range=True,
                ),
                FunctionWrapperDouble(
                    np.moveaxis, input=True, target=False, source=-1, destination=0
                ),
                FunctionWrapperDouble(normalize_01),
            ]
        )

        dataset = SegmentationDataSet(
            inputs=inputs, targets=targets, transform=transforms_test
        )

        return DataLoader(dataset=dataset, batch_size=1, shuffle=False)
