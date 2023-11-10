import torch

# Hyperparameters etc.
LOCAL = True
NOTEBOOK = False
LEARNING_RATE = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'cpu'
BATCH_SIZE = 1
NUM_EPOCHS = 2
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
TEST_MODEL=False
DATASET = 'lung'
LOADERS = None
if DATASET == 'carvana':
  TRAIN_IMG_DIR = "data/train_images/"
  TRAIN_MASK_DIR = "data/train_masks/"
  VAL_IMG_DIR = "data/val_images/"
  VAL_MASK_DIR = "data/val_masks/"
else: 
  TRAIN_IMG_DIR = "segmentation/train/image/"
  TRAIN_MASK_DIR = "segmentation/train/mask/"

if LOCAL:
    DATASET_DIR = "./input"
else:
    DATASET_DIR="/kaggle/input/carvana-image-masking-png"
