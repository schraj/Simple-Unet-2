import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from src.model_api.utils import save_predictions_as_imgs, save_checkpoint, load_checkpoint, check_accuracy
from src.model.unet import UNET
import src.config as h
from src.lung.lung_image_loader import LungImageLoader
from src.model_api.lifecycle import ModelLifecycle
from src.visualization.visualizer import Visualizer
from src.model.losses import BCELossModule, DiceLoss, CombinedLoss

class Trainer:
    def __init__(self):
        self.model = UNET(in_channels=3, out_channels=1).to(h.DEVICE) 
        self.modelLifecyle = ModelLifecycle(self.model)
        
    def train_fn(self, loader, optimizer, loss_fn, scaler):
        if h.NOTEBOOK:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=h.DEVICE)
            targets = targets.float().unsqueeze(1).to(device=h.DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                predictions = self.model(data)
                loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

    def test(self, include_visualization):
        self.modelLifecyle.load_model()
        lung_image_loader = LungImageLoader()
        test_loader = lung_image_loader.test_loader

        score, preds_array = check_accuracy(test_loader, self.model, device=h.DEVICE)

        if (include_visualization):
            visualizer = Visualizer(lung_image_loader, self.model)
            visualizer.show_test_results(15, preds_array)

        return preds_array

    def train(self):
        if (h.DATASET == 'carvana'):
            from src.carvana.loaders import CarvanaLoaders
            dataset = CarvanaLoaders()
            train_loader = dataset.train_loader
            val_loader = dataset.val_loader
        else:
            from src.lung.lung_image_loader import LungImageLoader
            dataset = LungImageLoader()
            train_loader = dataset.train_loader
            val_loader = dataset.val_loader

        bCELossModule = BCELossModule()
        diceLoss = DiceLoss()
        
        combinedLoss = CombinedLoss([bCELossModule, diceLoss], [1, 1], h.DEVICE)   
        loss_fn = combinedLoss
        # loss_fn = nn.BCEWithLogitsLoss()
        regularOptimizer = optim.Adam(self.model.parameters(), lr=h.LEARNING_RATE)
        firstOptimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        secondOptimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)
    
        current_score, _ = check_accuracy(val_loader, self.model, device=h.DEVICE)
        scaler = torch.cuda.amp.GradScaler()
        score_array = []
        best_score = 0 
        for epoch in range(h.NUM_EPOCHS):
            print("Epoch ",epoch)
            # optimizer = firstOptimizer if epoch < 15 else secondOptimizer
            self.train_fn(train_loader, regularOptimizer, loss_fn, scaler)

            current_score, preds_array = check_accuracy(val_loader, self.model, device=h.DEVICE)
            score_array.append(current_score)        
            if epoch > 5 and current_score > best_score:
                best_score = current_score
                print("Saving model...")
                print("Dice score:", best_score)
                print("Epoch:", epoch)
                self.modelLifecyle.save_model()
                print("Model saved")

        self.modelLifecyle.save_model()
        print('scores:', score_array)
        return score_array
