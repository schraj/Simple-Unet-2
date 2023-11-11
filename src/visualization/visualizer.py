import matplotlib.pyplot as plt
from src.visualization.visualization_utils import add_mask

class Visualizer:
  data_loader = None
  model = None
  def __init__(self, data_loader, model):
    self.data_loader = data_loader  
    self.mode = model

  def show_test_results(self, count):
    self.model.eval()
    count = min(count, len(self.data_loader))
    ctr = 0
    for image, mask in self.data_loader:
      merged_image = add_mask(image, mask)
      preds = self.model(image)                                  
      merged_prediction = add_mask(image, preds)

      _, axs = plt.subplots(2, 2, figsize=(15, 8))

      axs[0, 0].set_title("Target")
      axs[0, 0].imshow(merged_image.permute(1, 2, 0))

      axs[0, 1].set_title("Prediction")
      axs[0, 1].imshow(merged_prediction)

      ctr += 1
      if ctr == count:
        break