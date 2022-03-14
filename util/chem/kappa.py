from rdkit import Chem
from rdkit.Chem import GraphDescriptors as desc


class CalcKappa:
    """
    #################################################################
    Calculation of molecular shape index for one bonded fragment

    ---->kappa1

    Usage:

        result=Kappa1(mol)

        Input: mol is a molecule object.

        Output: result is a numeric value.
    #################################################################
    """
    def __init__(self, is_alpha=False, n_bond=1):
        self.is_alpha = is_alpha
        self.n_bond = n_bond

    def __call__(self, mol, **kwargs):
        alpha = desc.HallKierAlpha(mol) if self.is_alpha else 0.0
        # P = mol.GetNumBonds(onlyHeavy=1) + alpha
        P = len(Chem.FindAllPathsOfLengthN(mol, self.n_bond)) + alpha
        A = mol.GetNumHeavyAtoms() + alpha + 1 - self.n_bond
        denom = P + alpha
        if denom:
            if A % 2 == 1 and self.n_bond == 3:
                kappa = (A) * (A + 1) ** 2 / denom ** 2
            else:
                kappa = (A) * (A - 1) ** 2 / denom ** 2
        else:
            kappa = 0.0
        return round(kappa, 3)


def Flexibility(mol):
    """
    #################################################################
    Calculation of Kier molecular flexibility index

    ---->phi

    Usage:

        result=Flexibility(mol)

        Input: mol is a molecule object.

        Output: result is a numeric value.
    #################################################################
    """
    kappa1 = CalcKappa(is_alpha=True, n_bond=1)(mol)
    kappa2 = CalcKappa(is_alpha=True, n_bond=1)(mol)
    A = mol.GetNumHeavyAtoms()
    phi = kappa1 * kappa2 / (A + 0.0)

    return round(phi, 3)
