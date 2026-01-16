"""Pseudo-label generation and filtering."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import label
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize


@dataclass
class PseudoLabeler:
    threshold: float = 0.8
    min_size: int = 64
    fill_holes: bool = True
    max_skel_components: Optional[int] = None
    ignore_index: int = 255

    def _filter_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool)
        if self.min_size > 0:
            mask = remove_small_objects(mask, self.min_size)
        if self.fill_holes:
            mask = remove_small_holes(mask, self.min_size)
        if self.max_skel_components is not None:
            skel = skeletonize(mask)
            labeled, num = label(skel)
            if num > self.max_skel_components:
                mask[:] = False
        return mask.astype(np.uint8)

    def generate(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pseudo labels and confidence map."""
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            conf = probs.squeeze(1)
            pred = (conf > self.threshold).long()
            pseudo = pred.clone()
            pseudo[conf <= self.threshold] = self.ignore_index

            pseudo_np = pseudo.cpu().numpy()
            for i in range(pseudo_np.shape[0]):
                mask = (pseudo_np[i] == 1)
                mask = self._filter_mask(mask)
                pseudo_np[i] = np.where(mask, 1, 0)
            pseudo = torch.from_numpy(pseudo_np).to(logits.device)
            return pseudo, conf

        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pseudo = pred.clone()
        pseudo[conf <= self.threshold] = self.ignore_index

        if logits.shape[1] > 2:
            return pseudo, conf

        pseudo_np = pseudo.cpu().numpy()
        for i in range(pseudo_np.shape[0]):
            mask = (pseudo_np[i] == 1)
            mask = self._filter_mask(mask)
            pseudo_np[i] = np.where(mask, 1, 0)
        pseudo = torch.from_numpy(pseudo_np).to(logits.device)
        return pseudo, conf
