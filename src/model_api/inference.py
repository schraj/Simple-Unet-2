import torch
from src.model.unet import UNET
from src.model_api.lifecycle import ModelLifecycle
import config as h

class Inference:
    def __init__(self):
      self.model = UNET(in_channels=3, out_channels=1).to(h.DEVICE) 
      self.modelLifecyle = ModelLifecycle(self.model)

    def predict(self, x):
        self.model.eval()
        x = x.to(h.DEVICE)
        with torch.no_grad():
            preds = self.model(x)
            preds = (preds > 0.5).float()
        self.model.train()
        return preds
