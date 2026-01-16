"""Exponential moving average for model parameters."""

from copy import deepcopy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

    def to(self, device: torch.device) -> None:
        self.shadow.to(device)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.shadow.load_state_dict(state_dict)
