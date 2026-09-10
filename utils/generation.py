"""Molecule generation shared by the DEG2MOL test and inference entry points."""

import csv
import gc
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from torchdiffeq import odeint
from tqdm import tqdm

from utils.runtime import empty_cuda_cache


class GuidedFlowODE(torch.nn.Module):
    """Classifier-free-guided vector field for the accepted conditional model."""

    def __init__(self, flow_model, condition: torch.Tensor, guidance_scale: float):
        super().__init__()
        self.flow_model = flow_model
        self.condition = condition
        self.guidance_scale = guidance_scale

    def forward(self, t, x):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        t_batch = t.reshape(1, 1).expand(x.size(0), 1)
        conditional = self.flow_model(x, t_batch, self.condition)
        null_condition = self.flow_model.null_condition.expand(x.size(0), -1)
        unconditional = self.flow_model(x, t_batch, null_condition)
        return unconditional + self.guidance_scale * (conditional - unconditional)


def integrate_flow(
    flow_model,
    initial: torch.Tensor,
    condition: torch.Tensor,
    solver: str,
    num_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    """Integrate the learned flow from Gaussian noise at t=0 to t=1."""
    ode_func = GuidedFlowODE(flow_model, condition, guidance_scale)
    if solver == "dopri5":
        times = torch.tensor([0.0, 1.0], device=initial.device, dtype=initial.dtype)
        return odeint(ode_func, initial, times, method="dopri5", atol=1e-5, rtol=1e-5)[-1]

    if num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    dt = 1.0 / num_steps
    state = initial
    for step in range(num_steps):
        t = torch.tensor(step * dt, device=state.device, dtype=state.dtype)
        if solver == "rk4":
            k1 = ode_func(t, state)
            k2 = ode_func(t + 0.5 * dt, state + 0.5 * dt * k1)
            k3 = ode_func(t + 0.5 * dt, state + 0.5 * dt * k2)
            k4 = ode_func(t + dt, state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        elif solver == "heun":
            k1 = ode_func(t, state)
            k2 = ode_func(t + dt, state + dt * k1)
            state = state + (dt / 2.0) * (k1 + k2)
        elif solver == "euler":
            state = state + dt * ode_func(t, state)
        else:
            raise ValueError(f"Unknown solver: {solver}")
    return state


def _condition_latent(deg_model, deg_values: torch.Tensor) -> torch.Tensor:
    output = deg_model(deg_values)
    if isinstance(output, tuple) and len(output) >= 2:
        return output[1]
    if torch.is_tensor(output):
        return output
    raise TypeError("DEGMON did not return a usable latent representation.")


def _valid_molecule(smiles: Optional[str]):
    if not smiles or smiles in {"None", "INVALID", "INtest"}:
        return None
    return Chem.MolFromSmiles(smiles)


@torch.no_grad()
def generate_molecules(
    flow_model,
    deg_model,
    scaf_vae,
    data_loader,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    generation_batch_size: int,
    latent_dim: int,
    solver: str,
    num_steps: int,
    guidance_scale: float,
    normalize_condition: bool,
    reference_smiles: Optional[List[str]] = None,
    save_embeddings: bool = False,
) -> Tuple[Dict, int, int]:
    """Generate molecules and save both the legacy pickle and a portable CSV."""
    if num_samples <= 0 or generation_batch_size <= 0:
        raise ValueError("Sample counts and generation batch size must be positive.")
    if reference_smiles is not None and len(reference_smiles) != len(data_loader.dataset):
        raise ValueError("Reference SMILES and DEG rows must have the same length.")

    output_dir.mkdir(parents=True, exist_ok=True)
    flow_model.eval()
    deg_model.eval()
    scaf_vae.eval()

    results: Dict[str, Dict[str, List[Optional[Chem.Mol]]]] = {}
    csv_rows = []
    embedding_names, embedding_values = [], []
    generated_count = valid_count = row_index = 0

    with tqdm(total=len(data_loader.dataset), desc="Generating molecules", unit="sample") as progress:
        for batch_index, (deg_values, sample_names) in enumerate(data_loader):
            deg_values = deg_values.to(device, non_blocking=True)
            conditions = _condition_latent(deg_model, deg_values)

            for local_index, sample_name in enumerate(sample_names):
                condition = conditions[local_index : local_index + 1]
                if normalize_condition:
                    condition = F.normalize(condition, p=2, dim=1)
                molecules = []
                sample_latents = []

                for start in range(0, num_samples, generation_batch_size):
                    chunk_size = min(generation_batch_size, num_samples - start)
                    repeated_condition = condition.repeat(chunk_size, 1)
                    initial = torch.randn(chunk_size, latent_dim, device=device)
                    final_latents = integrate_flow(
                        flow_model,
                        initial,
                        repeated_condition,
                        solver,
                        num_steps,
                        guidance_scale,
                    )
                    decoded = scaf_vae.frag_decoder.sample(
                        batch_size=chunk_size,
                        input_noise=final_latents,
                        output_smi=True,
                    ).get("smi", [])
                    if save_embeddings:
                        sample_latents.append(final_latents.cpu().numpy())

                    for sample_number, smiles in enumerate(decoded, start=start):
                        molecule = _valid_molecule(smiles)
                        molecules.append(molecule)
                        generated_count += 1
                        valid_count += molecule is not None
                        csv_rows.append(
                            {
                                "input_index": row_index,
                                "cmap_name": sample_name,
                                "sample_index": sample_number,
                                "smiles": Chem.MolToSmiles(molecule) if molecule is not None else "",
                                "valid": int(molecule is not None),
                            }
                        )

                    del initial, final_latents, repeated_condition, decoded
                    empty_cuda_cache(device)

                result_label = reference_smiles[row_index] if reference_smiles is not None else sample_name
                results[f"{result_label}_{row_index}"] = {"generated_mols": molecules}
                if save_embeddings and sample_latents:
                    embedding_names.append(sample_name)
                    embedding_values.append(
                        np.concatenate(sample_latents, axis=0).mean(axis=0).astype(np.float32)
                    )
                row_index += 1
                progress.update(1)

            del deg_values, conditions
            if batch_index % 10 == 0:
                gc.collect()
            empty_cuda_cache(device)

    with (output_dir / "generated_molecules_dict.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (output_dir / "generated_molecules.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_index", "cmap_name", "sample_index", "smiles", "valid"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    if save_embeddings and embedding_values:
        np.savez(
            output_dir / "compound_embeddings.npz",
            names=np.asarray(embedding_names),
            embeddings=np.asarray(embedding_values, dtype=np.float32),
        )

    return results, generated_count, valid_count
