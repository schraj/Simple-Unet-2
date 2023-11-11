import torch
import pathlib

MODEL_NAME = "lung_segmentation.pt"
MODEL_PATH = pathlib.Path.cwd() / 'saved_model' / MODEL_NAME

class ModelLifecycle:
  def __init__(self, model):
    self.model = model

  def save_model(self):
    torch.save(self.model.state_dict(), MODEL_PATH)

  def load_model(self):
    self.model.load_state_dict(torch.load(MODEL_PATH))