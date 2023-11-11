import os
import matplotlib.pyplot as plt
import torch
from skimage.io import imread
import src.lung.constants as c
from src.lung.data_utils import get_file_name
from src.visualization.visualization_utils import add_mask

def show_sample_montgomery_image():  
  base_file = get_file_name(c.SEGMENTATION_IMAGE_DIR, 'M')
  image_file = os.path.join(c.SEGMENTATION_IMAGE_DIR, base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_MASK_DIR, base_file)

  image = imread(image_file)
  mask_image = imread(mask_image_file, as_gray=True)
  image, mask_image= torch.from_numpy(image), torch.from_numpy(mask_image)
  image = image.permute(2, 0, 1)
  merged_image = add_mask(image, mask_image)

  _, axs = plt.subplots(2, 3, figsize=(15, 8))

  axs[0, 0].set_title("X-Ray")
  axs[0, 0].imshow(image.permute(1, 2, 0))

  axs[0, 1].set_title("Mask")
  axs[0, 1].imshow(mask_image)

  axs[0, 2].set_title("Merged")
  axs[0, 2].imshow(merged_image.permute(1, 2, 0))

  base_file = get_file_name(c.SEGMENTATION_TEST_DIR, 'M')
  filename, fileext = os.path.splitext(base_file)
  image_file = os.path.join(c.SEGMENTATION_TEST_DIR, base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_TEST_DIR, "%s_mask%s" % (filename, fileext))

  image = imread(image_file)
  mask_image = imread(mask_image_file, as_gray=True)
  image, mask_image= torch.from_numpy(image), torch.from_numpy(mask_image)
  image = image.permute(2, 0, 1)
  merged_image = add_mask(image, mask_image)

  axs[1, 0].set_title("X-Ray")
  axs[1, 0].imshow(image.permute(1, 2, 0))

  axs[1, 1].set_title("Mask")
  axs[1, 1].imshow(mask_image)

  axs[1, 2].set_title("Merged")
  axs[1, 2].imshow(merged_image.permute(1, 2, 0))

def show_sample_shenszhen_image():
  base_file = get_file_name(c.SEGMENTATION_IMAGE_DIR, 'C')

  image_file = os.path.join(c.SEGMENTATION_IMAGE_DIR, base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_MASK_DIR, base_file)

  image = imread(image_file)
  mask_image = imread(mask_image_file, as_gray=True)
  image, mask_image= torch.from_numpy(image), torch.from_numpy(mask_image)
  image = image.permute(2, 0, 1)
  merged_image = add_mask(image, mask_image)
                            
  _, axs = plt.subplots(2, 3, figsize=(15, 8))

  axs[0, 0].set_title("X-Ray")
  axs[0, 0].imshow(image.permute(1, 2, 0))

  axs[0, 1].set_title("Mask")
  axs[0, 1].imshow(mask_image)

  axs[0, 2].set_title("Merged")
  axs[0, 2].imshow(merged_image.permute(1, 2, 0))

  base_file = get_file_name(c.SEGMENTATION_TEST_DIR, 'C')
  image_file = os.path.join(c.SEGMENTATION_TEST_DIR, base_file)
  filename, fileext = os.path.splitext(base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_TEST_DIR, \
                                "%s_mask%s" % (filename, fileext))

  filename, fileext = os.path.splitext(base_file)
  image_file = os.path.join(c.SEGMENTATION_TEST_DIR, base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_TEST_DIR, \
                                "%s_mask%s" % (filename, fileext))
  image = imread(image_file)
  mask_image = imread(mask_image_file, as_gray=True)
  image, mask_image= torch.from_numpy(image), torch.from_numpy(mask_image)
  image = image.permute(2, 0, 1)
  merged_image = add_mask(image, mask_image)

  axs[1, 0].set_title("X-Ray")
  axs[1, 0].imshow(image.permute(1, 2, 0))

  axs[1, 1].set_title("Mask")
  axs[1, 1].imshow(mask_image)

  axs[1, 2].set_title("Merged")
  axs[1, 2].imshow(merged_image.permute(1, 2, 0))