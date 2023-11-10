import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import generate_train_test_dataset
from src.utils import get_loaders
import src.hyperparams as h

class CarvanaLoaders:
    train_loader: None
    val_loader: None
    def __init__(self):
      IMAGE_HEIGHT = 160  # 1280 originally
      IMAGE_WIDTH = 240  # 1918 originally

      self.train_transform = A.Compose(
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

      train_ds,test_ds = generate_train_test_dataset(h.DATASET_DIR,self.train_transform,self.val_transforms,count=1)
    
      self.train_loader, self.val_loader = get_loaders(
          train_ds,
          test_ds,
          h.BATCH_SIZE,
          h.NUM_WORKERS,
          h.PIN_MEMORY,
      )

