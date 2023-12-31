import torch

# Hyperparameters etc.
LOCAL = True
NOTEBOOK = False
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
DATASET = 'lung'
if LOCAL:
  NUM_EPOCHS = 3
  NUM_WORKERS = 0
  PIN_MEMORY = True
else: 
  NUM_EPOCHS = 30
  NUM_WORKERS = 2
  PIN_MEMORY = True    