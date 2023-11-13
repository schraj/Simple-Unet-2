import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from src.model_api.utils import get_loaders
import src.config as h
from src.lung.data_utils import get_file_list
from src.lung.custom_datasets import SegmentationDataSet
import src.lung.constants as c

train_size = 0.8
random_seed = 42

class LungImageLoader:
    train_loader: None
    val_loader: None
    test_loader: None
    test_inputs: None
    test_targets: None
    def __init__(self):
      IMAGE_HEIGHT = 512  # 1280 originally
      IMAGE_WIDTH = 512  # 1918 originally

      self.train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
      )

      self.val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
      )

      self.test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
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
          inputs=self.inputs_train, targets=self.targets_train, transform=self.train_transforms
      )

      self.dataset_valid = SegmentationDataSet(
          inputs=self.inputs_valid, targets=self.targets_valid, transform=self.val_transforms
      )
    
      self.test_files = get_file_list(c.SEGMENTATION_TEST_DIR)
      self.test_targets = [s for s in self.test_files if 'mask' in s]
      self.test_inputs =  [s for s in self.test_files if 'mask' not in s]

      self.dataset_test = SegmentationDataSet(
          inputs=self.test_inputs, targets=self.test_targets, transform=self.test_transforms
      )

      self.train_loader, self.val_loader, self.test_loader = get_loaders(
          self.dataset_train,
          self.dataset_valid,
          self.dataset_test,
          h.BATCH_SIZE,
          h.NUM_WORKERS,
          h.PIN_MEMORY,
      )
