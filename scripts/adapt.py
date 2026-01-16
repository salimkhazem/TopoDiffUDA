"""UDA adaptation script."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
import time
from pathlib import Path

import torch

from src.data.datamodule import DataModule
from src.models.model_factory import build_model
from src.trainers.diffusion_aug import DiffusionAugmenter
from src.trainers.uda_trainer import UDATrainer
from src.utils.config import load_yaml, merge_dicts, save_yaml
from src.utils.io import ensure_dir, write_env_snapshot
from src.utils.logger import setup_logging
from src.utils.paths import get_outputs_root, get_repo_root
from src.utils.seed import get_generator, seed_everything, seed_worker


def load_config(args) -> dict:
    base = load_yaml(Path("configs/default.yaml"))
    dataset_cfg = load_yaml(Path(f"configs/datasets/{args.dataset}.yaml"))
    model_cfg = load_yaml(Path(f"configs/models/{args.model}.yaml"))
    exp_cfg = load_yaml(Path(args.config))
    cfg = merge_dicts(base, dataset_cfg)
    cfg = merge_dicts(cfg, model_cfg)
    cfg = merge_dicts(cfg, exp_cfg)
    cfg["dataset"]["name"] = args.dataset
    cfg["model"]["name"] = args.model
    cfg["seed"] = args.seed
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_tb", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--topo_weight", type=float, default=None)
    parser.add_argument("--diffusion_prob", type=float, default=None)
    parser.add_argument("--diffusion_manifest", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args)
    if args.topo_weight is not None:
        cfg["uda"]["topo_weight"] = args.topo_weight
    if args.diffusion_prob is not None:
        cfg.setdefault("diffusion", {})["prob"] = args.diffusion_prob
    if args.diffusion_manifest is not None:
        cfg.setdefault("diffusion", {})["manifest_path"] = args.diffusion_manifest

    if cfg.get("diffusion", {}).get("manifest_path"):
        ds_name = cfg.get("dataset", {}).get("source", args.dataset)
        cfg["diffusion"]["manifest_path"] = cfg["diffusion"]["manifest_path"].replace("{dataset}", ds_name)
    seed_everything(
        cfg["seed"],
        deterministic=cfg["train"].get("deterministic", True),
        warn_only=cfg["train"].get("deterministic_warn_only", False),
    )

    if args.exp_name:
        cfg.setdefault("experiment", {})["name"] = args.exp_name
    exp_name = cfg.get("experiment", {}).get("name", Path(args.config).stem)
    output_dir = get_outputs_root() / "runs" / exp_name / args.dataset / args.model / str(args.seed)
    ensure_dir(output_dir)
    save_yaml(output_dir / "config.yaml", cfg)
    write_env_snapshot(output_dir, get_repo_root())

    logger = setup_logging(output_dir, name="adapt")
    logger.info(f"Output dir: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scheduler = None
    if cfg["train"].get("lr_step"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["train"]["lr_step"], gamma=0.1)

    datamodule = DataModule(cfg, seed=cfg["seed"])
    loaders = datamodule.get_uda_loaders()

    if cfg.get("diffusion", {}).get("use", False):
        manifest_path = Path(cfg["diffusion"]["manifest_path"])
        if manifest_path.exists():
            augmenter = DiffusionAugmenter(manifest_path, prob=cfg["diffusion"].get("prob", 1.0))
            source_dataset = augmenter.wrap(loaders["source"].dataset)
            loaders["source"] = torch.utils.data.DataLoader(
                source_dataset,
                batch_size=loaders["source"].batch_size,
                shuffle=True,
                num_workers=loaders["source"].num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=get_generator(cfg["seed"]),
            )
        else:
            logger.warning("Diffusion manifest not found; continuing without diffusion aug")

    tb_writer = None
    if args.log_tb:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=output_dir / "tb")

    if args.log_wandb:
        try:
            import wandb

            wandb.init(project="topo-diffuda", config=cfg, name=f"{exp_name}-{args.dataset}-{args.model}-{args.seed}")
        except Exception:
            logger.warning("wandb not available; continuing without it")

    trainer = UDATrainer(
        model,
        optimizer,
        device,
        cfg,
        output_dir,
        scheduler=scheduler,
        tb_writer=tb_writer,
        logger=logger,
    )

    start_epoch = 0
    last_ckpt = output_dir / "last.ckpt"
    if args.resume and last_ckpt.exists():
        try:
            ckpt = torch.load(last_ckpt, map_location=device)
        except Exception as exc:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            corrupt_path = last_ckpt.with_name(f"{last_ckpt.name}.corrupt.{timestamp}")
            try:
                last_ckpt.rename(corrupt_path)
                logger.warning(
                    "Failed to load checkpoint %s (%s). Renamed to %s and restarting.",
                    last_ckpt,
                    exc,
                    corrupt_path,
                )
            except OSError:
                logger.warning("Failed to load checkpoint %s (%s). Restarting from scratch.", last_ckpt, exc)
        else:
            model.load_state_dict(ckpt["model"])
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as exc:
                logger.warning(
                    "Optimizer state in %s not compatible (%s). Continuing with a fresh optimizer.",
                    last_ckpt,
                    exc,
                )
            try:
                trainer.scaler.load_state_dict(ckpt.get("scaler", trainer.scaler.state_dict()))
            except Exception as exc:
                logger.warning(
                    "Scaler state in %s not compatible (%s). Continuing with a fresh scaler.",
                    last_ckpt,
                    exc,
                )
            if trainer.ema and ckpt.get("ema"):
                trainer.ema.load_state_dict(ckpt["ema"])
            start_epoch = ckpt.get("epoch", 0)
            logger.info(f"Resumed from {last_ckpt} at epoch {start_epoch}")

    trainer.fit(loaders, start_epoch=start_epoch, epochs=cfg["train"]["epochs"])

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
