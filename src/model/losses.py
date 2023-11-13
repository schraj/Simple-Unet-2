import torch
from typing import Optional, Sequence, Union
import torch.nn as nn

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
        weight: Optional[Sequence[float]] = None,
        device: Optional[torch.device] = None,
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

def dice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    dice,
    weight: float = 1.0,
    eps: float = 0.0001,
    smooth: float = 0.0,
):
    return dice(probs, target, weight, eps, smooth)

class BCELossModule(torch.nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if weight is None:
            weight = torch.tensor(1.0)
        self.register_buffer("weight", weight)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.BCEWithLogitsLoss(weight=self.weight)(output, target) 

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
        self.dice = dice_loss
        if weight is None:
            weight = torch.tensor(1.0)
        self.register_buffer("weight", weight)
        self.smooth = smooth

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = self.softmax(output)
        return self.dice(
            probs=probs, target=target, weight=self.weight, smooth=self.smooth
        )
