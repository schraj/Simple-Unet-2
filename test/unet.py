import torch
from src.model.unet import UNET

def test():
    x = torch.randn((16, 3, 161, 161))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape