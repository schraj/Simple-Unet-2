import torchvision
import os

from src.model_api.processor import Processor
import src.lung.constants as c
processor = Processor()
# processor.run_training(update_data=False)
# processor.run_test(False)

# prediction = processor.predict(os.path.join("./images/TorsoXray.png"))
prediction = processor.predict('/Users/jeschrader/code-port/simple-unet-2/segmentation/train/image/CHNCXR_0061_0.png')
torchvision.utils.save_image(
    prediction, f"saved_images/prediction.png"
)

# import numpy as np
# from skimage.io import imread
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# # image = np.array(Image.open('/Users/jeschrader/code-port/simple-unet-2/lung_images/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png/MCUCXR_0001_0.png').convert("RGB"), dtype=np.float32)
# image = np.array(Image.open('/Users/jeschrader/code-port/simple-unet-2/segmentation/train/image/CHNCXR_0061_0.png').convert("RGB"), dtype=np.float32)

# #image = imread('/Users/jeschrader/code-port/simple-unet-2/lung_images/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png/MCUCXR_0001_0.png')
# transforms = A.Compose(
# [
#     A.Resize(height=512, width=512),
#     A.Normalize(
#         mean=[0.0, 0.0, 0.0],
#         std=[1.0, 1.0, 1.0],
#         max_pixel_value=255.0,
#     ),
#     ToTensorV2(),
# ])      
# augmentations = transforms(image=image)
# transformed = augmentations['image']
# print(transformed)
