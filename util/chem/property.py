from rdkit.Chem import Crippen
from rdkit.Chem import MolSurf as MS
import math


class Property:
    def __init__(self, keys):
        self.keys = keys
        self.props = {
            'LogP': Crippen.MolLogP,
            'MR': Crippen.MolMR,
            'LabuteASA': MS.pyLabuteASA,
            'TPSA': MS.TPSA,
            'Hy': CalculateHydrophilicityFactor,
            'UI': CalculateUnsaturationIndex
        }

    def __call__(self, mol):
        return self.props[self.keys](mol)


def CalculateUnsaturationIndex(mol):
    """
    #################################################################
    Calculation of unsaturation index.

    ---->UI

    Usage:

        result=CalculateUnsaturationIndex(mol)

        Input: mol is a molecule object.

        Output: result is a numeric value.
    #################################################################
    """
    nd = sum([1 if b.GetBondType() == 2 else 0 for b in mol.GetBonds()])
    nt = sum([1 if b.GetBondType() == 3 else 0 for b in mol.GetBonds()])
    na = sum([1 if b.GetBondType() == 12 else 0 for b in mol.GetBonds()])
    res = math.log((1 + nd + nt + na), 2)

    return round(res, 3)


def CalculateHydrophilicityFactor(mol):
    """
    #################################################################
    Calculation of hydrophilicity factor. The hydrophilicity

    index is described in more detail on page 225 of the

    Handbook of Molecular Descriptors (Todeschini and Consonni 2000).

    ---->Hy

    Usage:

        result=CalculateHydrophilicityFactor(mol)

        Input: mol is a molecule object.

        Output: result is a numeric value.
    #################################################################
    """
    nheavy = mol.GetNumAtoms(onlyHeavy=1)
    nc = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            nc = nc + 1
    nhy = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7 or atom.GetAtomicNum() == 8 or atom.GetAtomicNum() == 16:
            atomn = atom.GetNeighbors()
            for i in atomn:
                if i.GetAtomicNum() == 1:
                    nhy = nhy + 1

    res = (1 + nhy) * math.log((1 + nhy), 2) + nc * (1.0 / nheavy * math.log(1.0 / nheavy, 2)) + math.sqrt(
        (nhy + 0.0) / (nheavy ^ 2))
    return round(res, 3)



