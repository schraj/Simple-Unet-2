#utils
import torch
from torch.utils.data import DataLoader
import torchvision
from torchmetrics.classification import Dice

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    running_manual_dice_score = 0
    running_tm_dice_score = 0
    loader_len = len(loader)
    model.eval()
    dice=Dice().to(device)
    preds_array = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))

            preds = (preds > 0.5).float()
            preds_array.append(preds)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            indiv_dice_score = dice_coefficient(preds, y)
            running_manual_dice_score += indiv_dice_score
            target = (y == 1)
            # print('manual:', indiv_dice_score)
            tm_dice = dice(preds, target)
            running_tm_dice_score += tm_dice
            # print('tm:', tm_dice)

    average_manual_dice_score = (running_manual_dice_score / loader_len)
    average_tm_dice_score = (running_tm_dice_score / loader_len)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    # print(f"Manual Dice score: {average_manual_dice_score}")
    print(f"TM Dice score: {average_tm_dice_score}")
    model.train()
    return average_tm_dice_score, preds_array

# This is not correct, giving values above 1
def dice_coefficient(pred, target, epsilon=1e-6):
    """
    Compute the Dice coefficient.

    :param pred: Tensor of predicted segmentation.
    :param target: Tensor of ground truth segmentation.
    :param epsilon: Small value to avoid division by zero.
    :return: Dice coefficient.
    """

    # Flatten the tensors to ensure it works for various input shapes
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Compute Dice coefficient
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice

def save_predictions_as_imgs(
    loader, model, folder="saved_images", device="cuda"
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
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
        else:
            break
        count -= 1
        
    model.train()