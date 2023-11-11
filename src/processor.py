

import os
import torch
import numpy as np
from glob import glob
import src.data.constants as c
from src.data.prepare_data import DataPreparer
from src.trainer import Trainer

class Processor:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def run_training(self, update_data = True):
        if (update_data): 
            data_preparer = DataPreparer()
            data_preparer.prepare_data()

        # Check number of image files
        train_files = glob(os.path.join(c.SEGMENTATION_IMAGE_DIR, "*.png"))
        test_files = glob(os.path.join(c.SEGMENTATION_TEST_DIR, "*.png"))
        mask_files = glob(os.path.join(c.SEGMENTATION_MASK_DIR, "*.png"))
        print(len(train_files), len(test_files), len(mask_files))

        trainer = Trainer()
        trainer.train()
    def run_test(self):
        trainer = Trainer()
        trainer.test()



