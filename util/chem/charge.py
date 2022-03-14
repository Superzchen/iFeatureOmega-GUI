from rdkit import Chem
from rdkit.Chem import rdPartialCharges as GMCharge

import numpy as np

iter_step = 12


class CalcElementCharge:
    def __init__(self, num_atom=6, method=np.min):
        self.n_atom = num_atom
        self.method = method

    def __call__(self, mol, **kwargs):
        Hmol = Chem.AddHs(mol)
        GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
        res = []
        for atom in Hmol.GetAtoms():
            if atom.GetAtomicNum() == self.n_atom or self.n_atom == 0:
                res.append(float(atom.GetProp('_GasteigerCharge')))
        if res == []:
            return 0
        else:
            return self.method(np.array(res))


def sqsum(arr):
    return np.sum(np.square(arr))


def abs_sum(arr):
    return np.sum(np.absolute(arr))


def abs_mean(arr):
    return np.mean(np.absolute(arr))


def relp_sum(arr):
    return max(arr) / sum(arr[arr > 0])


def reln_sum(arr):
    return max(arr) / sum(arr[arr < 0])


def p_sum(arr):
    return sum(arr[arr > 0])


def p_mean(arr):
    return np.mean(arr[arr > 0])


def n_sum(arr):
    return sum(arr[arr < 0])


def n_mean(arr):
    return np.mean(arr[arr < 0])


def CalcLocalDipoleIndex(mol):
    """
    #################################################################
    Calculation of local dipole index (D)

    -->LDI

    Usage:

        result=CalculateLocalDipoleIndex(mol)

        Input: mol is a molecule object.

        Output: result is a numeric value.
    #################################################################
    """

    GMCharge.ComputeGasteigerCharges(mol, iter_step)
    res = []
    for atom in mol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))
    cc = [np.absolute(res[x.GetBeginAtom().GetIdx()] - res[x.GetEndAtom().GetIdx()]) for x in mol.GetBonds()]
    B = len(mol.GetBonds())

    return round(sum(cc) / B, 3)


def CalcSubmolPolarityPara(mol):
    """
    #################################################################
    Calculation of submolecular polarity parameter(SPP)

    -->SPP

    Usage:

        result=CalculateSubmolPolarityPara(mol)

        Input: mol is a molecule object.

        Output: result is a numeric value.
    #################################################################
    """
    maxP = CalcElementCharge(0, np.max)
    maxN = CalcElementCharge(0, np.min)
    return round(maxP(mol) - maxN(mol), 3)


