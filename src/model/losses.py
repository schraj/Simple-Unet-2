import torch
from typing import Optional, Sequence, Union
import torch.nn as nn
from torchmetrics.classification import Dice
import src.config as h

class CombinedLoss(torch.nn.Module):
    """Defines a loss function as a weighted sum of combinable loss criteria.
    Args:
        criteria: List of loss criterion modules that should be combined.
        weight: Weight assigned to the individual loss criteria (in the same
            order as ``criteria``).
        device: The device on which the loss should be computed. This needs
            to be set to the device that the loss arguments are allocated on.
    """

    def __init__(
        self,
        criteria,
        weight,
        device,
    ):
        super().__init__()
        self.criteria = torch.nn.ModuleList(criteria)
        self.device = device
        if weight is None:
            weight = torch.ones(len(criteria))
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            assert weight.shape == (len(criteria),)
        self.register_buffer("weight", weight.to(self.device))

    def forward(self, *args):
        loss = torch.tensor(0.0, device=self.device)
        for crit, weight in zip(self.criteria, self.weight):
            loss += weight * crit(*args)
        return loss

class BCELossModule(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_with_logits_loss(output, target)

class DiceLoss(torch.nn.Module):
    def __init__(
        self,
        apply_softmax: bool = True,
        weight: Optional[torch.Tensor] = None,
        smooth: float = 0.0,
    ):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)
        self.dice = Dice().to(h.DEVICE)
        if weight is None:
            weight = torch.tensor(1.0)
        self.register_buffer("weight", weight)
        self.smooth = smooth

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = self.softmax(output)
        target = (target== 1)
        return self.dice(probs, target)
