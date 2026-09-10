"""Generate molecules from a user-provided DEG table.

Adapted from 04_inference.py for the accepted Gated-MLP/DEGMON-AE model. The
input table is only read; generated files are written under --output-dir.
"""

import argparse

from models.DEGMON.DEG_AE import GO_Autoencoder
from models.flow.MLP import GatedConditionalFlowMLP
from utils.data import create_deg_loader, read_deg_table
from utils.generation import generate_molecules
from utils.runtime import (
    load_gene_order,
    load_model_weights,
    load_scafvae,
    project_path,
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

    frame = read_deg_table(args.input_path, args.max_samples)
    if frame.empty:
        raise ValueError("The input DEG table has no rows.")
    loader = create_deg_loader(
        frame,
        genes,
        "inference",
        args.batch_size,
        args.num_workers,
    )

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

    input_stem = project_path(args.input_path).stem
    output_dir = project_path(args.output_dir or f"outputs/inference/{args.split}/{input_stem}")
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
    )
    print(
        f"Generated {generated_count} molecules for {len(results)} DEG rows; "
        f"{valid_count} were RDKit-valid. Results: {output_dir}"
    )


def build_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-path", default="data/inference/KO/extra_test.feather")
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--model-checkpoint", default=None)
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
