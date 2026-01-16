"""Domain adversarial components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.grl = GradientReversal()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            features = torch.mean(features, dim=(2, 3))
        x = self.grl(features)
        return self.classifier(x)


def domain_adv_loss(discriminator: DomainDiscriminator, features: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
    logits = discriminator(features)
    return F.cross_entropy(logits, domain_labels)
