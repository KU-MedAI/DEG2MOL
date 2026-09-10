"""Generate and evaluate molecules on a DEG2MOL random or scaffold test split.

Adapted from 03_2_evaluate.py for the accepted Gated-MLP/DEGMON-AE model.
"""

import argparse
import json

import pandas as pd

from models.DEGMON.DEG_AE import GO_Autoencoder
from models.flow.MLP import GatedConditionalFlowMLP
from utils.data import create_deg_loader, read_deg_table, split_data_root
from utils.evaluation import evaluate_generated_mols
from utils.generation import generate_molecules
from utils.runtime import (
    load_gene_order,
    load_model_weights,
    load_scafvae,
    project_path,
    require_path,
    seed_everything,
    select_device,
)


DEGMON_DIMS = [10280, 2011, 1614, 1075]


def checkpoint_for_split(split, override=None):
    return override or f"checkpoints/{split}/best_model.pt"


def main(args):
    seed_everything(args.seed)
    device = select_device(args.device)
    genes = load_gene_order(args.gene_list_path)
    if len(genes) != DEGMON_DIMS[0]:
        raise ValueError(
            f"The accepted DEGMON checkpoint expects {DEGMON_DIMS[0]} genes, got {len(genes)}."
        )

    data_root = split_data_root(args.split, args.data_root)
    train_path = require_path(data_root / "train.feather", f"{args.split} training data")
    train_smiles = pd.read_feather(train_path, columns=["cmap_name", "canonical_smiles"])
    test_frame = read_deg_table(str(data_root / "test.feather"))
    if not args.skip_scaffold_filter:
        scaffold_dir = require_path(
            project_path(args.task_path) / "scaf",
            "ScafVAE scaffold features",
            directory=True,
        )
        train_available = train_smiles["cmap_name"].map(
            lambda name: (scaffold_dir / f"{name}.npz").is_file()
        )
        test_available = test_frame["cmap_name"].map(
            lambda name: (scaffold_dir / f"{name}.npz").is_file()
        )
        print(
            f"Scaffold filter removed {(~train_available).sum()} train and "
            f"{(~test_available).sum()} test rows."
        )
        train_smiles = train_smiles.loc[train_available].reset_index(drop=True)
        test_frame = test_frame.loc[test_available].reset_index(drop=True)
    if args.target_compounds:
        test_frame = test_frame[
            test_frame["cmap_name"].isin(args.target_compounds)
        ].reset_index(drop=True)
    if args.max_eval_samples is not None:
        test_frame = test_frame.head(args.max_eval_samples).copy()
    if test_frame.empty:
        raise ValueError("No test rows remain after filtering.")
    if "canonical_smiles" not in test_frame:
        raise ValueError("Train and test tables must contain a 'canonical_smiles' column.")

    flow_model = GatedConditionalFlowMLP(
        embedding_dim=args.latent_dim,
        condition_dim=args.latent_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        combine_method=args.combine_method,
        dropout=args.dropout,
    ).to(device)
    flow_checkpoint = checkpoint_for_split(args.split, args.model_checkpoint)
    load_model_weights(flow_model, flow_checkpoint, device, "DEG2MOL checkpoint")

    deg_model = GO_Autoencoder(dims=DEGMON_DIMS, latent_dim=args.latent_dim).to(device)
    load_model_weights(deg_model, args.deg_checkpoint, device, "DEGMON checkpoint")
    scafvae, _ = load_scafvae(args.scafvae_checkpoint, device)

    loader = create_deg_loader(
        test_frame,
        genes,
        f"{args.split} test",
        args.batch_size,
        args.num_workers,
    )
    output_dir = project_path(args.output_dir or f"outputs/test/{args.split}")
    results, generated_count, valid_count = generate_molecules(
        flow_model=flow_model,
        deg_model=deg_model,
        scaf_vae=scafvae,
        data_loader=loader,
        device=device,
        output_dir=output_dir,
        num_samples=args.num_samples,
        generation_batch_size=args.generation_batch_size,
        latent_dim=args.latent_dim,
        solver=args.solver,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        normalize_condition=args.normalize_condition,
        reference_smiles=test_frame["canonical_smiles"].astype(str).tolist(),
        save_embeddings=args.save_embeddings,
    )
    metrics = evaluate_generated_mols(
        results_dict=results,
        train_smiles=train_smiles["canonical_smiles"].dropna().astype(str).tolist(),
    )
    summary = {
        "split": args.split,
        "model_checkpoint": str(project_path(flow_checkpoint)),
        "test_rows": len(test_frame),
        "molecules_requested_per_row": args.num_samples,
        "molecules_returned": generated_count,
        "valid_molecules": valid_count,
        "solver": args.solver,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation_results.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--model-checkpoint", default=None)
    parser.add_argument("--data-root", default=None, help="Override data/splits/<split>")
    parser.add_argument("--task-path", default="data/scafvae/deg2mol_64dim")
    parser.add_argument(
        "--skip-scaffold-filter",
        action="store_true",
        help="Evaluate every table row instead of matching the accepted ScafVAE cohort",
    )
    parser.add_argument(
        "--gene-list-path",
        default="data/BP/gene_attribute_matrix_overlap_with_L1000.csv",
    )
    parser.add_argument(
        "--deg-checkpoint",
        default="checkpoints/DEGMON_AE_best_model.pth",
    )
    parser.add_argument("--scafvae-checkpoint", default="checkpoints/ScafVAE.chk")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--combine-method", choices=["sum", "concat", "cross_attn"], default="sum")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--solver", choices=["euler", "heun", "rk4", "dopri5"], default="euler")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--normalize-condition", action="store_true")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--target-compounds", nargs="+", default=None)
    parser.add_argument("--no-save-embeddings", dest="save_embeddings", action="store_false")
    parser.set_defaults(save_embeddings=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
