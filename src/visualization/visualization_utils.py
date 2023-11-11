import torch
from torchvision.utils import draw_segmentation_masks

def add_mask(image, mask_image):
    mask_bool = (mask_image!=0)
    ret = draw_segmentation_masks(image, mask_bool, 0.7)

    return ret

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    count = 3
    for idx, (x, y) in enumerate(loader):
        if count >0:
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        else:
            break
        count -= 1