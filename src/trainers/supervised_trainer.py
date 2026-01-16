"""Supervised training loop."""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.losses.ce_dice import CEDiceLoss
from src.utils.distributed import is_main_process
from src.utils.logger import MetricsLogger
from src.utils.meters import AverageMeter
from src.utils.metrics import compute_segmentation_metrics


class SupervisedTrainer:
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
        self.tb_writer = tb_writer
        self.logger = logger
        self.progress_bar = cfg["train"].get("progress_bar", True)

    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        loss_meter = AverageMeter()
        grad_accum = self.cfg["train"].get("grad_accum_steps", 1)
        self.optimizer.zero_grad()

        use_tqdm = self.progress_bar and is_main_process()
        progress = loader
        if use_tqdm:
            progress = tqdm(
                loader,
                desc=f"Train {epoch + 1}/{self.cfg['train']['epochs']}",
                leave=False,
            )

        for step, batch in enumerate(progress):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            with autocast(enabled=self.cfg["train"].get("amp", False)):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks) / grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            loss_meter.update(loss.item() * grad_accum, images.size(0))
            if use_tqdm:
                progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if self.scheduler is not None:
            self.scheduler.step()
        return loss_meter.avg

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

        # Aggregate
        agg = {}
        if not metrics:
            return agg
        for key in metrics[0].keys():
            agg[key] = float(sum(m[key] for m in metrics) / len(metrics))
        return agg

    def fit(self, train_loader, val_loader, start_epoch: int, epochs: int) -> Dict[str, float]:
        best_metric = -1.0
        best_path = self.output_dir / "best.ckpt"
        last_path = self.output_dir / "last.ckpt"
        history = {"train_loss": [], "val_mdice": []}

        for epoch in range(start_epoch, epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)
            val_score = val_metrics.get("mdice", 0.0)

            history["train_loss"].append(train_loss)
            history["val_mdice"].append(val_score)

            self.metrics_logger.log({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/loss", train_loss, epoch)
                for key, value in val_metrics.items():
                    self.tb_writer.add_scalar(f"val/{key}", value, epoch)

            if self.logger is not None:
                msg_parts = [f"Epoch {epoch + 1}/{epochs}", f"train_loss={train_loss:.4f}"]
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
                },
                last_path,
            )

        self.metrics_logger.summarize({"best_mdice": best_metric})
        return {"best_mdice": best_metric}
