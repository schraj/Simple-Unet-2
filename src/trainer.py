#model_training

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import save_predictions_as_imgs, save_checkpoint, load_checkpoint, check_accuracy
from src.unet import UNET
import src.hyperparams as h

class Trainer:
    def __init__(self):
        pass

    def train_fn(self, loader, model, optimizer, loss_fn, scaler):
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
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())


    def process(self):
        if (h.DATASET == 'carvana'):
            from src.carvana.loaders import CarvanaLoaders
            dataset = CarvanaLoaders()
            train_loader = dataset.train_loader
            val_loader = dataset.val_loader
        else:
            from src.data.loaders import LungImageLoaders
            dataset = LungImageLoaders()
            train_loader = dataset.train_loader
            val_loader = dataset.val_loader
 
        model = UNET(in_channels=3, out_channels=1).to(h.DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=h.LEARNING_RATE)
        
        if h.LOAD_MODEL:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
            
        # if h.TEST_MODEL:
        #     print("Only testing...")
        #     check_accuracy(val_loader, model, device=h.DEVICE)
        #     save_predictions_as_imgs(
        #         val_loader, model, folder="saved_images/", device=h.DEVICE
        #         )
        #     return
            
        check_accuracy(val_loader, model, device=h.DEVICE)
        scaler = torch.cuda.amp.GradScaler()
        
        best_score = 0
        for epoch in range(h.NUM_EPOCHS):
            print("Epoch ",epoch)
            self.train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # save model
            # checkpoint = {
            #     "state_dict": model.state_dict(),
            #     "optimizer":optimizer.state_dict(),
            # }
            # save_checkpoint(checkpoint)

            # check accuracy
            current_score = check_accuracy(val_loader, model, device=h.DEVICE)
            
            if current_score > best_score:
                best_score = current_score 
                # save_checkpoint(checkpoint,"best_checkpoint_pth.tar")
                # save_predictions_as_imgs(
                # val_loader, model, folder="saved_images/", device=h.DEVICE
                # )
            # print some examples to a folder
            
