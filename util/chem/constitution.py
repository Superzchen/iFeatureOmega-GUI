from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as desc
import numpy as np


class PathsOfLengthN:
    def __init__(self, n):
        self.n = n

    def __call__(self, mol, **kwargs):
        paths = Chem.FindAllPathsOfLengthN(mol, self.n)
        return len(paths)


class TotalAtom:
    def __init__(self, is_weight=False):
        self.is_weight = is_weight

    def __call__(self, mol, **kwargs):
        mol = Chem.AddHs(mol)
        n_atoms = len(mol.GetAtoms())
        if self.is_weight:
            return desc.ExactMolWt(mol) / n_atoms
        return n_atoms


class FragCounter:
    def __init__(self, element: str) -> None:
        """
        Args:
            element: element to count within a molecule
        """
        self.element = element

    def __call__(self, mol, **kwargs):
        """
        Count the number of atoms of a given type.
        Args:
            mol: molecule
        Returns:
            The number of atoms of the given type.
        """
        # if the molecule contains H atoms, they may be implicit, so add them
        sub = Chem.MolFromSmarts(self.element)
        if self.element == '[H]':
            mol = Chem.AddHs(mol)
        score = len(mol.GetSubstructMatches(sub))
        return score

