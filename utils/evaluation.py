"""Basic molecule-generation metrics used by the accepted evaluation script."""

import itertools
from typing import Dict, List, Optional, Set

from rdkit import Chem


def calculate_validity(molecules: List[Optional[Chem.Mol]]) -> float:
    if not molecules:
        return 0.0
    return sum(molecule is not None for molecule in molecules) / len(molecules)


def calculate_uniqueness(molecules: List[Chem.Mol]) -> float:
    if not molecules:
        return 0.0
    unique = {Chem.MolToSmiles(molecule) for molecule in molecules}
    return len(unique) / len(molecules)


def calculate_novelty(molecules: List[Chem.Mol], train_smiles: Set[str]) -> float:
    if not molecules:
        return 0.0
    novel = sum(Chem.MolToSmiles(molecule) not in train_smiles for molecule in molecules)
    return novel / len(molecules)


def evaluate_generated_mols(results_dict: Dict, train_smiles: List[str]) -> Dict[str, float]:
    """Compute validity, uniqueness, and novelty over all generated molecules."""
    molecules = list(
        itertools.chain.from_iterable(
            entry["generated_mols"] for entry in results_dict.values()
        )
    )
    valid_molecules = [molecule for molecule in molecules if molecule is not None]
    canonical_train = {
        Chem.MolToSmiles(molecule)
        for smiles in train_smiles
        for molecule in [Chem.MolFromSmiles(smiles)]
        if molecule is not None
    }
    return {
        "validity": calculate_validity(molecules),
        "uniqueness": calculate_uniqueness(valid_molecules),
        "novelty": calculate_novelty(valid_molecules, canonical_train),
    }
