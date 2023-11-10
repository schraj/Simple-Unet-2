import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    


from sklearn.model_selection import train_test_split

def generate_train_test_dataset(data_dir,train_transforms,test_transforms , count=1.0, prefix="train_"):
    image_dir = os.path.join(data_dir,prefix+"images")
    mask_dir = os.path.join(data_dir,prefix+"masks")
    images = os.listdir(image_dir)
    masks = os.listdir(mask_dir)
    whole_images = [ os.path.join(image_dir,filename) for filename in images[0:int(len(images)*count)]]
    whole_masks = [os.path.join(mask_dir,filename) for filename in masks[0:int(len(masks)*count)]]
    whole_images=whole_images[:2]
    whole_masks=whole_masks[:2]
    train_x,test_x,train_y,test_y = train_test_split(whole_images,whole_masks)
    train = CarvanaDataset(train_x,train_y,train_transforms)
    test = CarvanaDataset(test_x,test_y,test_transforms)
    return train,test
 