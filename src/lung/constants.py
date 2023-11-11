import os
import src.config as h

if h.LOCAL:
  INPUT_DIR = os.path.join("./lung_images")
else: 
  INPUT_DIR = os.path.join("/kaggle/input")

SEGMENTATION_SOURCE_DIR = os.path.join(INPUT_DIR, "pulmonary-chest-xray-abnormalities")
SHENZHEN_TRAIN_INTER_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "ChinaSet_AllFiles")
SHENZHEN_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "ChinaSet_AllFiles","ChinaSet_AllFiles")
SHENZHEN_IMAGE_DIR = os.path.join(SHENZHEN_TRAIN_DIR, "CXR_png")
SHENZHEN_TOP_MASK_DIR = os.path.join(INPUT_DIR, "shcxr-lung-mask",)
SHENZHEN_MASK_INTER_DIR = os.path.join(INPUT_DIR, "shcxr-lung-mask", "mask")
SHENZHEN_MASK_DIR = os.path.join(INPUT_DIR, "shcxr-lung-mask", "mask", "mask")
MONTGOMERY_TRAIN_INTER_DIR = os.path.join(SEGMENTATION_SOURCE_DIR,"Montgomery")
MONTGOMERY_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR,"Montgomery", "MontgomerySet")
MONTGOMERY_IMAGE_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "CXR_png")
MONTGOMERY_MASK_DIR = os.path.join(MONTGOMERY_TRAIN_DIR,"ManualMask")
MONTGOMERY_LEFT_MASK_DIR = os.path.join(MONTGOMERY_TRAIN_DIR,"ManualMask", "leftMask")
MONTGOMERY_RIGHT_MASK_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "ManualMask", "rightMask")

if h.LOCAL:
  SEGMENTATION_DIR = os.path.join("./segmentation")
else: 
  SEGMENTATION_DIR = os.path.join("/kaggle/working/simple-unet-2/segmentation")

SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, "test")
SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, "train")
SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "image")
SEGMENTATION_MASK_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "mask")