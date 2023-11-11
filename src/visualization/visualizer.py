import torch
import matplotlib.pyplot as plt
from skimage.io import imread
from src.visualization.visualization_utils import add_mask

class Visualizer:
  lung_image_loader = None
  model = None
  def __init__(self, lung_image_loader, model):
    self.lung_image_loader = lung_image_loader  
    self.model = model

  def show_test_results(self, count, preds_array):
    self.model.eval()
    count = min(count, len(self.lung_image_loader.test_loader))
    ctr = 0
    for image_file, mask_image_file in zip(self.lung_image_loader.test_inputs, self.lung_image_loader.test_targets):
      preds = preds_array[ctr]
      image = imread(image_file)
      mask_image = imread(mask_image_file, as_gray=True)
      image, mask_image= torch.from_numpy(image), torch.from_numpy(mask_image)
      image = image.permute(2, 0, 1)
      merged_image = add_mask(image, mask_image)
      preds = torch.squeeze(preds)
      merged_prediction = add_mask(image, preds)

      _, axs = plt.subplots(2, 2, figsize=(15, 8))

      axs[0, 0].set_title("Target")
      axs[0, 0].imshow(merged_image.permute(1, 2, 0))

      axs[0, 1].set_title("Prediction")
      axs[0, 1].imshow(merged_prediction)

      ctr += 1
      if ctr == count:
        break