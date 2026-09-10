"""Train the accepted DEG2MOL conditional flow-matching architecture.

This public entry point is adapted from 02_2_flow_DEG_condition_ema.py. All
inputs are read-only and every relative path is anchored at the repository root.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:  # optional experiment tracking
    wandb = None

try:
    from ScafVAE.utils.dataset_utils import ScafDataset, collate_ligand
except ImportError as exc:
    raise ImportError(
        "ScafVAE is unavailable. Initialize and install third_party/ScafVAE "
        "as described in README.md."
    ) from exc

from models.DEGMON.DEG_AE import GO_Autoencoder
from models.flow.MLP import GatedConditionalFlowMLP
from utils.data import split_data_root
from utils.runtime import (
    load_gene_order,
    load_model_weights,
    load_scafvae,
    project_path,
    require_path,
    seed_everything,
    select_device,
)
from utils.training_utils import AverageMeter, create_run_directory, save_checkpoint, save_config


DEGMON_DIMS = [10280, 2011, 1614, 1075]


class EMAManager:
    """Maintain the exponential moving average saved by the accepted runs."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].mul_(self.decay).add_(parameter, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        self.backup = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(self.backup[name])
        self.backup = {}


class DEGandScafDataset(ScafDataset):
    """Index-aligned DEG/ScafVAE dataset with one-to-many molecule support."""

    def __init__(self, frame, genes, mode, scafvae_args, task_path):
        super().__init__(
            mode,
            scafvae_args,
            data_path=str(task_path / "feat"),
            data_list=str(task_path),
            scaf_path=str(task_path / "scaf"),
            name="DEG2MOL",
        )
        if "cmap_name" not in frame.columns:
            raise ValueError(f"{mode} data must contain a 'cmap_name' column.")
        missing = [gene for gene in genes if gene not in frame.columns]
        if missing:
            raise ValueError(f"{mode} data is missing {len(missing)} required genes: {missing[:10]}")
        self.deg_values = torch.from_numpy(
            frame.loc[:, genes].to_numpy(dtype=np.float32, copy=True)
        )
        self.full_id_list = frame["cmap_name"].astype(str).tolist()
        self.data_list = self.full_id_list
        self.sub_data_list = self.full_id_list

    def __getitem__(self, index):
        molecule = super().__getitem__(index)
        expected = self.full_id_list[index]
        if molecule["idx"] != expected:
            raise RuntimeError(
                f"Dataset alignment failed at row {index}: "
                f"ScafVAE='{molecule['idx']}', DEG='{expected}'."
            )
        return molecule, self.deg_values[index]

    def __len__(self):
        return len(self.deg_values)


def collate_deg_and_ligand(batch):
    molecules = collate_ligand([item[0] for item in batch])
    deg_values = torch.stack([item[1] for item in batch], dim=0)
    return molecules, deg_values


def filter_available_scaffolds(frame: pd.DataFrame, scaffold_dir: Path, split_name: str):
    available = frame["cmap_name"].map(lambda name: (scaffold_dir / f"{name}.npz").is_file())
    missing = int((~available).sum())
    print(f"{split_name}: removed {missing} rows without a scaffold feature file")
    return frame.loc[available].reset_index(drop=True)


def create_dataloaders(args, genes, scafvae_args, device):
    data_root = split_data_root(args.split, args.data_root)
    train_path = require_path(data_root / "train.feather", f"{args.split} training data")
    valid_path = require_path(data_root / "valid.feather", f"{args.split} validation data")
    task_path = require_path(args.task_path, "ScafVAE task assets", directory=True)
    require_path(task_path / "feat", "ScafVAE molecular features", directory=True)
    scaffold_dir = require_path(task_path / "scaf", "ScafVAE scaffold features", directory=True)

    train_frame = filter_available_scaffolds(
        pd.read_feather(train_path), scaffold_dir, "train"
    )
    valid_frame = filter_available_scaffolds(
        pd.read_feather(valid_path), scaffold_dir, "valid"
    )
    train_dataset = DEGandScafDataset(train_frame, genes, "train", scafvae_args, task_path)
    valid_dataset = DEGandScafDataset(valid_frame, genes, "val", scafvae_args, task_path)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_deg_and_ligand,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_options),
        DataLoader(valid_dataset, shuffle=False, **loader_options),
    )


def load_frozen_models(args, genes, device):
    if len(genes) != DEGMON_DIMS[0]:
        raise ValueError(
            f"The accepted DEGMON checkpoint expects {DEGMON_DIMS[0]} genes, got {len(genes)}."
        )
    deg_model = GO_Autoencoder(dims=DEGMON_DIMS, latent_dim=args.latent_dim).to(device)
    load_model_weights(deg_model, args.deg_checkpoint, device, "DEGMON checkpoint")
    deg_model.eval()
    for parameter in deg_model.parameters():
        parameter.requires_grad = False

    scafvae, scafvae_args = load_scafvae(
        args.scafvae_checkpoint, device, need_dataset_args=True
    )
    for parameter in scafvae.parameters():
        parameter.requires_grad = False
    return deg_model, scafvae, scafvae_args


def molecular_latent(scafvae, molecule_batch):
    encoded = scafvae.frag_encoder(molecule_batch)
    for key in ("noise_mean", "mu", "mean", "noise"):
        if key in encoded:
            return encoded[key]
    raise KeyError("ScafVAE encoder output has no molecular latent field.")


def train_one_epoch(flow_model, deg_model, scafvae, loader, optimizer, device, args, scaler, ema):
    flow_model.train()
    meter = AverageMeter()
    progress = tqdm(loader, desc="Train", ncols=100)
    amp_enabled = args.use_amp and device.type == "cuda"

    for molecule_batch, deg_values in progress:
        deg_values = deg_values.to(device, non_blocking=True)
        molecule_batch = {
            key: value.to(device, non_blocking=True)
            for key, value in molecule_batch.items()
            if torch.is_tensor(value)
        }
        batch_size = deg_values.size(0)
        with autocast(enabled=amp_enabled):
            with torch.no_grad():
                condition = deg_model(deg_values)[1]
                target = molecular_latent(scafvae, molecule_batch)

            source = torch.randn_like(target)
            distances = torch.cdist(source.view(batch_size, -1), target.view(batch_size, -1)).pow(2)
            source_rows, target_rows = linear_sum_assignment(distances.detach().cpu().numpy())
            source_rows = torch.as_tensor(source_rows, device=device)
            target_rows = torch.as_tensor(target_rows, device=device)
            source = source[source_rows]
            target = target[target_rows]
            condition = condition[target_rows]

            time_points = torch.sigmoid(torch.randn(batch_size, 1, device=device))
            time_points = time_points * (1.0 - 2e-5) + 1e-5
            interpolated = (1.0 - time_points) * source + time_points * target
            velocity = target - source
            if torch.rand((), device=device) < args.cfg_drop_prob:
                condition = flow_model.null_condition.expand(batch_size, -1)
            loss = F.mse_loss(flow_model(interpolated, time_points, condition), velocity)

        optimizer.zero_grad()
        if amp_enabled:
            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), args.gradient_clip)
            optimizer.step()
        if ema is not None:
            ema.update()
        meter.update(loss.item(), batch_size)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return meter.avg


@torch.no_grad()
def validate(flow_model, deg_model, scafvae, loader, device, args, ema):
    if ema is not None:
        ema.apply_shadow()
    flow_model.eval()
    meter = AverageMeter()
    amp_enabled = args.use_amp and device.type == "cuda"
    try:
        for molecule_batch, deg_values in loader:
            deg_values = deg_values.to(device, non_blocking=True)
            molecule_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in molecule_batch.items()
                if torch.is_tensor(value)
            }
            with autocast(enabled=amp_enabled):
                condition = deg_model(deg_values)[1]
                target = molecular_latent(scafvae, molecule_batch)
                source = torch.randn_like(target)
                time_points = torch.rand(target.size(0), 1, device=device)
                interpolated = (1.0 - time_points) * source + time_points * target
                velocity = target - source
                loss = F.mse_loss(flow_model(interpolated, time_points, condition), velocity)
            meter.update(loss.item(), target.size(0))
    finally:
        if ema is not None:
            ema.restore()
    return meter.avg


def setup_wandb(args):
    if not args.use_wandb:
        return None
    if wandb is None:
        raise ImportError("Install the optional 'wandb' package to use --use-wandb.")
    return wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        tags=args.wandb_tags,
        config=vars(args),
    )


def main(args):
    seed_everything(args.seed)
    device = select_device(args.device)
    genes = load_gene_order(args.gene_list_path)
    output_root = project_path(args.save_dir or f"outputs/training/{args.split}")
    run_dir = create_run_directory(output_root, args.run_name)
    save_config(args, run_dir)
    tracker = setup_wandb(args)

    deg_model, scafvae, scafvae_args = load_frozen_models(args, genes, device)
    train_loader, valid_loader = create_dataloaders(args, genes, scafvae_args, device)
    flow_model = GatedConditionalFlowMLP(
        embedding_dim=args.latent_dim,
        condition_dim=args.latent_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        combine_method=args.combine_method,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        flow_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        if args.use_scheduler
        else None
    )
    ema = EMAManager(flow_model, args.ema_decay) if args.use_ema else None
    scaler = GradScaler(enabled=args.use_amp and device.type == "cuda")
    best_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Training {args.split} split on {device}; outputs: {run_dir}")
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(
            flow_model, deg_model, scafvae, train_loader, optimizer, device, args, scaler, ema
        )
        valid_loss = validate(flow_model, deg_model, scafvae, valid_loader, device, args, ema)
        if scheduler is not None:
            scheduler.step()
        is_best = valid_loss < best_loss
        if is_best:
            best_loss = valid_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        metrics = {"loss": valid_loss}
        if tracker is not None:
            tracker.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": valid_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        if (epoch + 1) % args.save_interval == 0 or is_best:
            if ema is not None:
                ema.apply_shadow()
            save_checkpoint(
                epoch, flow_model, optimizer, scheduler, metrics, run_dir, is_best, max_keep=3
            )
            if ema is not None:
                ema.restore()
        print(
            f"Epoch {epoch + 1:03d} | Train {train_loss:.5f} | "
            f"Valid {valid_loss:.5f} | LR {optimizer.param_groups[0]['lr']:.2e}"
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            break
    if tracker is not None:
        tracker.finish()


def boolean_flag(parser, dest, default, help_text):
    option = dest.replace("_", "-")
    negative_option = option[4:] if option.startswith("use-") else option
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{option}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{negative_option}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def build_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--data-root", default=None, help="Override data/splits/<split>")
    parser.add_argument("--task-path", default="data/scafvae/deg2mol_64dim")
    parser.add_argument(
        "--gene-list-path",
        default="data/BP/gene_attribute_matrix_overlap_with_L1000.csv",
    )
    parser.add_argument(
        "--deg-checkpoint",
        default="checkpoints/DEGMON_AE_best_model.pth",
    )
    parser.add_argument("--scafvae-checkpoint", default="checkpoints/ScafVAE.chk")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--combine-method", choices=["sum", "concat", "cross_attn"], default="sum")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.3)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    boolean_flag(parser, "use_ema", True, "Use EMA weights for validation and checkpoints")
    boolean_flag(parser, "use_amp", True, "Use CUDA automatic mixed precision")
    boolean_flag(parser, "use_scheduler", True, "Use cosine warm-restart scheduling")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--run-name", default=f"flow_final_{time.strftime('%Y%m%d-%H%M')}")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="deg2mol-flow")
    parser.add_argument("--wandb-tags", nargs="+", default=["Target-mu", "OT", "GatedMLP"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
