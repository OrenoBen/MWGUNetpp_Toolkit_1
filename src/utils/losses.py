import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        # logits: (B,1,H,W), targets: (B,1,H,W)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2,3)) + self.eps
        den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.eps
        dice = 1 - (num / den)
        return dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + (1-self.bce_weight) * self.dice(logits, targets)
