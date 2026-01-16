"""Segmentation metrics."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: Optional[int] = None) -> np.ndarray:
    pred = pred.astype(np.int64)
    target = target.astype(np.int64)
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    idx = num_classes * target + pred
    cm = np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def iou_from_confusion(cm: np.ndarray) -> Tuple[np.ndarray, float]:
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = tp + fp + fn + 1e-8
    iou = tp / denom
    miou = float(np.nanmean(iou))
    return iou, miou


def dice_from_confusion(cm: np.ndarray) -> Tuple[np.ndarray, float]:
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn + 1e-8
    dice = (2 * tp) / denom
    mdice = float(np.nanmean(dice))
    return dice, mdice


def pixel_accuracy(cm: np.ndarray) -> float:
    correct = np.diag(cm).sum()
    total = cm.sum() + 1e-8
    return float(correct / total)


def cldice_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    eps = 1e-8
    skel_pred = skeletonize(pred_bin > 0)
    skel_gt = skeletonize(gt_bin > 0)
    tprec = (skel_pred & (gt_bin > 0)).sum() / (skel_pred.sum() + eps)
    tsens = (skel_gt & (pred_bin > 0)).sum() / (skel_gt.sum() + eps)
    return float((2 * tprec * tsens) / (tprec + tsens + eps))


def boundary_fscore(pred_bin: np.ndarray, gt_bin: np.ndarray, tolerance: int = 2) -> float:
    pred_b = find_boundaries(pred_bin, mode="inner")
    gt_b = find_boundaries(gt_bin, mode="inner")
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    dt_pred = distance_transform_edt(~pred_b)
    dt_gt = distance_transform_edt(~gt_b)
    match_pred = (dt_gt <= tolerance) & pred_b
    match_gt = (dt_pred <= tolerance) & gt_b
    precision = match_pred.sum() / (pred_b.sum() + 1e-8)
    recall = match_gt.sum() / (gt_b.sum() + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Compute ECE using max prob and predicted label."""
    probs = probs.reshape(-1, probs.shape[-1])
    targets = targets.reshape(-1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return {"ece": float(ece)}


def topology_proxies(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
    skel_pred = skeletonize(pred_bin > 0)
    skel_gt = skeletonize(gt_bin > 0)
    length_ratio = (skel_pred.sum() + 1e-8) / (skel_gt.sum() + 1e-8)
    comp_pred = int(np.max(label_components(pred_bin > 0)))
    comp_gt = int(np.max(label_components(gt_bin > 0)))
    return {"skeleton_length_ratio": float(length_ratio), "components_pred": comp_pred, "components_gt": comp_gt}


def label_components(bin_mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label

    labeled, _ = label(bin_mask)
    return labeled


def compute_segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
    probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    iou, miou = iou_from_confusion(cm)
    dice, mdice = dice_from_confusion(cm)
    acc = pixel_accuracy(cm)
    metrics = {
        "miou": miou,
        "mdice": mdice,
        "pixel_acc": acc,
    }
    for i in range(num_classes):
        metrics[f"iou_{i}"] = float(iou[i])
        metrics[f"dice_{i}"] = float(dice[i])

    if num_classes == 2:
        pred_bin = pred == 1
        gt_bin = target == 1
        metrics["cldice"] = cldice_score(pred_bin, gt_bin)
        metrics["boundary_f"] = boundary_fscore(pred_bin, gt_bin)
        metrics.update(topology_proxies(pred_bin, gt_bin))

    if probs is not None:
        metrics.update(expected_calibration_error(probs, target))

    return metrics
