#model_training

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
from torchmetrics.classification import Dice


class Trainer:
    def __init__(self):
        self.model = UNET(in_channels=3, out_channels=1).to(h.DEVICE) 
        self.modelLifecyle = ModelLifecycle(self.model)

    # def combined_loss(self, loss_fn, dice, predictions, targets):
    #     loss_fn = nn.BCEWithLogitsLoss()
    #     bce_loss = loss_fn(predictions, targets)
    #     tm_dice = 1 - dice(predictions, targets)
    #     scale_factor = abs(bce_loss)/tm_dice
    #     scaled_dice = tm_dice * scale_factor
    #     loss = scaled_dice + bce_loss
    #     return loss
        
    def train_fn(self, loader, optimizer, loss_fn, scaler):
        if h.NOTEBOOK:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        loop = tqdm(loader)

        dice=Dice().to(h.DEVICE)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=h.DEVICE)
            targets = targets.float().unsqueeze(1).to(device=h.DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                predictions = self.model(data)
                # loss = loss_fn(predictions, targets)
                preds = torch.sigmoid(predictions)
                preds = (preds > 0.5).float()
                target = (targets == 1)
                loss = 1 - dice(preds, target)

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
 
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=h.LEARNING_RATE)
            
        current_score, _ = check_accuracy(val_loader, self.model, device=h.DEVICE)
        scaler = torch.cuda.amp.GradScaler()
        score_array = []
        best_score = 0 
        for epoch in range(h.NUM_EPOCHS):
            print("Epoch ",epoch)
            self.train_fn(train_loader, optimizer, loss_fn, scaler)

            current_score, preds_array = check_accuracy(val_loader, self.model, device=h.DEVICE)
            score_array.append(current_score)        
            if epoch > 15 and current_score > best_score:
                best_score = current_score
                print("Saving model...")
                print("Dice score:", best_score)
                print("Epoch:", epoch)
                self.modelLifecyle.save_model()
                print("Model saved")

        self.modelLifecyle.save_model()
        print('scores:', score_array)
        return score_array
