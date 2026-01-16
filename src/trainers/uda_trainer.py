"""UDA training loop with diffusion augmentation and topology loss."""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.losses.ce_dice import CEDiceLoss
from src.losses.consistency import consistency_loss
from src.losses.domain_adv import DomainDiscriminator, domain_adv_loss
from src.losses.loss_factory import compute_topology_consistency, compute_topology_loss
from src.models.ema import EMA
from src.trainers.pseudo_labeler import PseudoLabeler
from src.utils.distributed import is_main_process
from src.utils.logger import MetricsLogger
from src.utils.meters import AverageMeter
from src.utils.metrics import compute_segmentation_metrics


class UDATrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: Dict,
        output_dir: Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        tb_writer=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.output_dir = output_dir
        self.scheduler = scheduler
        self.scaler = scaler or GradScaler(enabled=cfg["train"].get("amp", False))
        self.loss_fn = CEDiceLoss(ignore_index=cfg["dataset"].get("ignore_index"))
        self.metrics_logger = MetricsLogger(output_dir)
        self.pseudo_labeler = PseudoLabeler(
            threshold=cfg["uda"].get("pseudo_threshold", 0.8),
            min_size=cfg["uda"].get("pseudo_min_size", 64),
            fill_holes=cfg["uda"].get("pseudo_fill_holes", True),
            max_skel_components=cfg["uda"].get("pseudo_max_skel_components"),
            ignore_index=cfg["dataset"].get("ignore_index", 255),
        )
        self.ema = None
        if cfg["uda"].get("use_ema", False):
            self.ema = EMA(model, decay=cfg["uda"].get("ema_decay", 0.999))
            self.ema.to(device)
        self.discriminator = None
        self.tb_writer = tb_writer
        self.logger = logger
        self.progress_bar = cfg["train"].get("progress_bar", True)

    def _pseudo_loss(self, logits: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            mask = pseudo != self.pseudo_labeler.ignore_index
            if mask.sum() > 0:
                ce = F.binary_cross_entropy_with_logits(
                    logits.squeeze(1)[mask], (pseudo[mask] == 1).float()
                )
            else:
                ce = torch.tensor(0.0, device=logits.device)
            probs = torch.sigmoid(logits)
            pseudo_bin = (pseudo == 1).float().unsqueeze(1)
            mask = (pseudo != self.pseudo_labeler.ignore_index).float().unsqueeze(1)
            intersection = (probs * pseudo_bin * mask).sum()
            cardinality = ((probs + pseudo_bin) * mask).sum()
            dice = 1.0 - (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        else:
            ce = F.cross_entropy(logits, pseudo, ignore_index=self.pseudo_labeler.ignore_index)
            probs = torch.softmax(logits, dim=1)
            num_classes = probs.shape[1]
            pseudo_onehot = F.one_hot(pseudo.clamp_min(0), num_classes).permute(0, 3, 1, 2).float()
            mask = (pseudo != self.pseudo_labeler.ignore_index).float().unsqueeze(1)
            intersection = (probs * pseudo_onehot * mask).sum(dim=(0, 2, 3))
            cardinality = ((probs + pseudo_onehot) * mask).sum(dim=(0, 2, 3))
            dice = 1.0 - (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
            dice = dice.mean()
        return ce + dice

    def train_epoch(self, loaders: Dict[str, torch.utils.data.DataLoader], epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        grad_accum = self.cfg["train"].get("grad_accum_steps", 1)
        self.optimizer.zero_grad()

        source_loader = loaders["source"]
        target_loader = loaders["target"]
        target_strong_loader = loaders["target_strong"]

        use_tqdm = self.progress_bar and is_main_process()
        iterator = zip(source_loader, target_loader, target_strong_loader)
        if use_tqdm:
            total = min(len(source_loader), len(target_loader), len(target_strong_loader))
            iterator = tqdm(
                iterator,
                total=total,
                desc=f"UDA {epoch + 1}/{self.cfg['train']['epochs']}",
                leave=False,
            )

        for step, (src, tgt, tgt_s) in enumerate(iterator):
            src_img = src["image"].to(self.device)
            src_mask = src["mask"].to(self.device)
            tgt_img = tgt["image"].to(self.device)
            tgt_img_s = tgt_s["image"].to(self.device)

            with autocast(enabled=self.cfg["train"].get("amp", False)):
                src_logits = self.model(src_img)
                sup_loss = self.loss_fn(src_logits, src_mask)

                if self.ema is not None:
                    with torch.no_grad():
                        weak_logits = self.ema.shadow(tgt_img)
                else:
                    weak_logits = self.model(tgt_img)

                pseudo, _ = self.pseudo_labeler.generate(weak_logits.detach())
                strong_logits = self.model(tgt_img_s)
                pseudo_loss = self._pseudo_loss(strong_logits, pseudo)

                topo_loss = compute_topology_loss(strong_logits, pseudo)
                topo_cons = compute_topology_consistency(strong_logits, (pseudo == 1).float())

                cons_loss = consistency_loss(weak_logits, strong_logits)

                total_loss = sup_loss
                total_loss += self.cfg["uda"].get("pseudo_weight", 1.0) * pseudo_loss
                total_loss += self.cfg["uda"].get("topo_weight", 0.1) * topo_loss
                total_loss += self.cfg["uda"].get("topo_cons_weight", 0.05) * topo_cons
                total_loss += self.cfg["uda"].get("consistency_weight", 0.1) * cons_loss

                if self.cfg["uda"].get("domain_adv", False):
                    src_feat = self.model.get_features(src_img)
                    tgt_feat = self.model.get_features(tgt_img)
                    if self.discriminator is None:
                        self.discriminator = DomainDiscriminator(src_feat.shape[1]).to(self.device)
                        self.optimizer.add_param_group({"params": self.discriminator.parameters()})
                    domain_labels = torch.cat(
                        [torch.zeros(src_feat.size(0)), torch.ones(tgt_feat.size(0))], dim=0
                    ).long().to(self.device)
                    feats = torch.cat([src_feat, tgt_feat], dim=0)
                    adv_loss = domain_adv_loss(self.discriminator, feats, domain_labels)
                    total_loss += self.cfg["uda"].get("domain_adv_weight", 0.1) * adv_loss

            total_loss = total_loss / grad_accum
            self.scaler.scale(total_loss).backward()

            if (step + 1) % grad_accum == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema is not None:
                    self.ema.update(self.model)

            loss_meter.update(total_loss.item() * grad_accum, src_img.size(0))
            if use_tqdm:
                iterator.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if self.scheduler is not None:
            self.scheduler.step()
        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        metrics = []
        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            logits = self.model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).long().squeeze(1)
                probs_np = torch.cat([1 - probs, probs], dim=1).permute(0, 2, 3, 1).cpu().numpy()
                num_classes = 2
            else:
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)
                probs_np = probs.permute(0, 2, 3, 1).cpu().numpy()
                num_classes = probs.shape[1]

            pred_np = pred.cpu().numpy()
            mask_np = masks.cpu().numpy()
            for i in range(pred_np.shape[0]):
                metrics.append(
                    compute_segmentation_metrics(
                        pred_np[i],
                        mask_np[i],
                        num_classes=num_classes,
                        ignore_index=self.cfg["dataset"].get("ignore_index"),
                        probs=probs_np[i],
                    )
                )

        agg = {}
        if not metrics:
            return agg
        for key in metrics[0].keys():
            agg[key] = float(sum(m[key] for m in metrics) / len(metrics))
        return agg

    def fit(self, loaders: Dict[str, torch.utils.data.DataLoader], start_epoch: int, epochs: int) -> Dict[str, float]:
        best_metric = -1.0
        best_path = self.output_dir / "best.ckpt"
        last_path = self.output_dir / "last.ckpt"

        for epoch in range(start_epoch, epochs):
            train_metrics = self.train_epoch(loaders, epoch)
            val_metrics = self.evaluate(loaders["target_val"])
            val_score = val_metrics.get("mdice", 0.0)
            self.metrics_logger.log({"epoch": epoch, **train_metrics, **val_metrics})
            if self.tb_writer is not None:
                for key, value in {**train_metrics, **val_metrics}.items():
                    self.tb_writer.add_scalar(key, value, epoch)

            if self.logger is not None:
                msg_parts = [f"Epoch {epoch + 1}/{epochs}", f"loss={train_metrics.get('loss', 0.0):.4f}"]
                if "mdice" in val_metrics:
                    msg_parts.append(f"val_mdice={val_metrics['mdice']:.4f}")
                if "miou" in val_metrics:
                    msg_parts.append(f"val_miou={val_metrics['miou']:.4f}")
                self.logger.info(" | ".join(msg_parts))

            if val_score > best_metric:
                best_metric = val_score
                torch.save({"model": self.model.state_dict()}, best_path)

            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "epoch": epoch + 1,
                    "best_metric": best_metric,
                    "ema": self.ema.state_dict() if self.ema else None,
                },
                last_path,
            )

        self.metrics_logger.summarize({"best_mdice": best_metric})
        return {"best_mdice": best_metric}
