"""Minimal checks for topology losses."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch

from src.losses.topology import cldice_loss, soft_skeletonize


def main() -> None:
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = (torch.rand(2, 1, 64, 64) > 0.7).float()

    skel = soft_skeletonize(pred, iters=5)
    assert skel.shape == pred.shape
    loss = cldice_loss(pred, target, iters=5)
    assert torch.isfinite(loss).item()

    loss.backward()
    assert pred.grad is not None

    # Toy: make pred closer to target and expect lower loss
    pred2 = target.clone().detach().requires_grad_(True)
    loss2 = cldice_loss(pred2, target, iters=5)
    print(f"Loss random: {loss.item():.4f}, loss perfect: {loss2.item():.4f}")
    assert loss2 < loss

    print("Topology loss checks passed.")


if __name__ == "__main__":
    main()
