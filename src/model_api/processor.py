

import os
import torch
from glob import glob
import src.lung.constants as c
from src.lung.prepare_data import DataPreparer
from src.model_api.trainer import Trainer

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
        score_array = trainer.train()
        return score_array
    def run_test(self, include_visualization):
        trainer = Trainer()
        preds_array = trainer.test(include_visualization)
        return preds_array

    def predict(self):
        trainer = Trainer()
        preds_array = trainer.predict()
        return preds_array
