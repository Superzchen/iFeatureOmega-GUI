from rdkit.Chem.EState import Fingerprinter as ESFP
from rdkit import Chem
from rdkit.Chem.EState import GetPrincipleQuantumNumber

from . import AtomTypes as ATEstate
import numpy as np


def _CalculateEState(mol, skipH=1):
    """
    #################################################################
    Get the EState value of each atom in a molecule
    #################################################################
    """
    mol = Chem.AddHs(mol)
    if skipH == 1:
        mol = Chem.RemoveHs(mol)
    tb1 = Chem.GetPeriodicTable()
    nAtoms = mol.GetNumAtoms()
    Is = np.zeros(nAtoms, np.float)
    for i in range(nAtoms):
        at = mol.GetAtomWithIdx(i)
        atNum = at.GetAtomicNum()
        d = at.GetDegree()
        if d > 0:
            h = at.GetTotalNumHs()
            dv = tb1.GetNOuterElecs(atNum) - h
            # dv=np.array(_AtomHKDeltas(at),'d')
            N = GetPrincipleQuantumNumber(atNum)
            Is[i] = (4.0 / (N * N) * dv + 1) / d
    dists = Chem.GetDistanceMatrix(mol, useBO=0, useAtomWts=0)
    dists += 1
    accum = np.zeros(nAtoms, np.float)
    for i in range(nAtoms):
        for j in range(i + 1, nAtoms):
            p = dists[i, j]
            if p < 1e6:
                temp = (Is[i] - Is[j]) / (p * p)
                accum[i] += temp
                accum[j] -= temp
    res = accum + Is
    return res


def CalcHeavyAtomEState(mol, **kwargs):
    """
    #################################################################
    The sum of the EState indices over all non-hydrogen atoms

    -->Shev
    #################################################################
    """
    return sum(_CalculateEState(mol))


def _CalculateAtomEState(mol, AtomicNum=6):
    """
    #################################################################
    **Internal used only**

    The sum of the EState indices over all atoms
    #################################################################
    """
    nAtoms = mol.GetNumAtoms()
    Is = np.zeros(nAtoms, np.float)
    Estate = _CalculateEState(mol)

    for i in range(nAtoms):
        at = mol.GetAtomWithIdx(i)
        atNum = at.GetAtomicNum()
        if atNum == AtomicNum:
            Is[i] = Estate[i]
    res = sum(Is)

    return res


def CalcCAtomEState(mol, **kwargs):
    """
    #################################################################
    The sum of the EState indices over all C atoms

    -->Scar
    #################################################################
    """
    return _CalculateAtomEState(mol, AtomicNum=6)


def CalculateHalogenEState(mol, **kwargs):
    """
    #################################################################
    The sum of the EState indices over all Halogen atoms

    -->Shal
    #################################################################
    """

    Nf = _CalculateAtomEState(mol, AtomicNum=9)
    Ncl = _CalculateAtomEState(mol, AtomicNum=17)
    Nbr = _CalculateAtomEState(mol, AtomicNum=35)
    Ni = _CalculateAtomEState(mol, AtomicNum=53)

    return Nf + Ncl + Nbr + Ni


def CalculateHeteroEState(mol, **kwargs):
    """
    #################################################################
    The sum of the EState indices over all hetero atoms

    -->Shet
    #################################################################
    """

    Ntotal = sum(_CalculateEState(mol))
    NC = _CalculateAtomEState(mol, AtomicNum=6)
    NH = _CalculateAtomEState(mol, AtomicNum=1)

    return Ntotal - NC - NH


def CalcAverageEState(mol, **kwargs):
    """
    #################################################################
    The sum of the EState indices over all non-hydrogen atoms

    divided by the number of non-hydrogen atoms.

    -->Save
    #################################################################
    """
    return np.mean(_CalculateEState(mol))


def CalcMaxEState(mol, **kwargs):
    """
    #################################################################
    Obtain the maximal Estate value in all atoms

    -->Smax
    #################################################################
    """
    return max(_CalculateEState(mol))


def CalcMinEState(mol, **kwargs):
    """
    #################################################################
    Obtain the minimal Estate value in all atoms

    -->Smin
    #################################################################
    """

    return min(_CalculateEState(mol))


def CalcDiffMaxMinEState(mol, **kwargs):
    """
    #################################################################
    The difference between Smax and Smin

    -->DS
    #################################################################
    """
    return max(_CalculateEState(mol)) - min(_CalculateEState(mol))


def CalcAtomTypeEState(mol, **kwargs):
    """
    #################################################################
    Calculation of sum of E-State value of specified atom type

    res---->list type
    #################################################################
    """
    AT = ATEstate.GetAtomLabel(mol)
    Estate = _CalculateEState(mol)
    res = []
    for i in AT:
        if i == []:
            res.append(0)
        else:
            res.append(sum([Estate[k] for k in i]))
    return res


def CalcEstateFingerprint(mol, **kwargs):
    """
    #################################################################
    The Calculation of EState Fingerprints.

    It is the number of times each possible atom type is hit.

    Usage:

        result=CalculateEstateFingerprint(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing 79 estate fragments.
    #################################################################
    """
    res = ESFP.FingerprintMol(mol)[0]
    return res


def CalcEstateValue(mol, **kwargs):
    """
    #################################################################
    The Calculate of EState Values.

    It is the sum of the Estate indices for atoms of each type.

    Usage:

        result=CalculateEstateValue(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing 79 estate values.
    #################################################################
    """
    res = ESFP.FingerprintMol(mol)[1]
    return res


def CalcMaxAtomTypeEState(mol, **kwargs):
    """
    #################################################################
    Calculation of maximum of E-State value of specified atom type

    res---->dict type

    Usage:

        result=CalculateMaxAtomTypeEState(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing 79 max estate values.
    #################################################################
    """
    AT = ATEstate.GetAtomLabel(mol)
    Estate = _CalculateEState(mol)
    res = []
    for i in AT:
        if i == []:
            res.append(0)
        else:
            res.append(max([Estate[k] for k in i]))
    return res


def CalcMinAtomTypeEState(mol):
    """
    #################################################################
    Calculation of minimum of E-State value of specified atom type

    res---->dict type

    Usage:

        result=CalculateMinAtomTypeEState(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing 79 min estate values.
    #################################################################
    """
    AT = ATEstate.GetAtomLabel(mol)
    Estate = _CalculateEState(mol)
    res = []
    for i in AT:
        if i == []:
            res.append(0)
        else:
            res.append(min([Estate[k] for k in i]))
    return res

