from rdkit.Chem import MolSurf
from rdkit.Chem.EState import EState_VSA as EVSA
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from .estate import CalcEstateFingerprint as EstateFingerprint
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit import DataStructs
import numpy as np


def CalcDaylightFingerprint(mol, **kwargs):
    """
    #################################################################
    Calculate Daylight-like fingerprint or topological fingerprint

    (128 bits).

    Usage:

        result=CalculateDaylightFingerprint(mol)

        Input: mol is a molecule object.

        Output: result is a tuple form. The first is the number of

        fingerprints. The second is a dict form whose keys are the

        position which this molecule has some substructure. The third

        is the DataStructs which is used for calculating the similarity.
    #################################################################
    """
    res = np.zeros(128)
    # NumFinger = 128
    bv = FingerprintMols.FingerprintMol(mol)
    DataStructs.ConvertToNumpyArray(bv, res)

    return res


def CalculateMACCSFingerprint(mol, **kwargs):
    """
    #################################################################
    Calculate MACCS keys (166 bits).

    Usage:

        result=CalculateMACCSFingerprint(mol)

        Input: mol is a molecule object.

        Output: result is a tuple form. The first is the number of

        fingerprints. The second is a dict form whose keys are the

        position which this molecule has some substructure. The third

        is the DataStructs which is used for calculating the similarity.
    #################################################################
    """
    res = np.zeros(166)
    bv = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(bv, res)

    return res


def CalcEstateFingerprint(mol, **kwargs):
    """
    #################################################################
    Calculate E-state fingerprints (79 bits).

    Usage:

        result=CalculateEstateFingerprint(mol)

        Input: mol is a molecule object.

        Output: result is a tuple form. The first is the number of

        fingerprints. The second is a dict form whose keys are the

        position which this molecule has some substructure. The third

        is the DataStructs which is used for calculating the similarity.
    #################################################################
    """
    NumFinger = 79
    res = {}
    temp = EstateFingerprint(mol, **kwargs)
    for i in temp:
        if temp[i] > 0:
            res[i[7:]] = 1

    return temp


def CalcAtomPairsFingerprint(mol, **kwargs):
    """
    #################################################################
    Calculate atom pairs fingerprints

    Usage:

        result=CalculateAtomPairsFingerprint(mol)

        Input: mol is a molecule object.

        Output: result is a tuple form. The first is the number of

        fingerprints. The second is a dict form whose keys are the

        position which this molecule has some substructure. The third

        is the DataStructs which is used for calculating the similarity.
    #################################################################
    """
    res = np.zeros(1)
    bv = Pairs.GetAtomPairFingerprint(mol)
    DataStructs.ConvertToNumpyArray(bv, res)
    return res


def CalculateTopologicalTorsionFingerprint(mol, **kwargs):
    """
    #################################################################
    Calculate Topological Torsion Fingerprints

    Usage:

        result=CalculateTopologicalTorsionFingerprint(mol)

        Input: mol is a molecule object.

        Output: result is a tuple form. The first is the number of

        fingerprints. The second is a dict form whose keys are the

        position which this molecule has some substructure. The third

        is the DataStructs which is used for calculating the similarity.
    #################################################################
    """
    res = Torsions.GetTopologicalTorsionFingerprint(mol)

    return res.GetLength(), res.GetNonzeroElements(), res


def CalculateMorganFingerprint(mol, **kwargs):
    """
    #################################################################
    Calculate Morgan

    Usage:

        result=CalculateMorganFingerprint(mol)

        Input: mol is a molecule object.

        radius is a radius.

        Output: result is a tuple form. The first is the number of

        fingerprints. The second is a dict form whose keys are the

        position which this molecule has some substructure. The third

        is the DataStructs which is used for calculating the similarity.
    #################################################################
    """
    res = AllChem.GetMorganFingerprint(mol, 2)

    return res.GetLength(), res.GetNonzeroElements(), res


def CalculateSimilarity(fp1, fp2, similarity="Tanimoto"):
    """
    #################################################################
    Calculate similarity between two molecules.

    Usage:

        result=CalculateSimilarity(fp1,fp2)

        Input: fp1 and fp2 are two DataStructs.

        Output: result is a similarity value.
    #################################################################
    """
    temp = DataStructs.similarityFunctions
    for i in temp:
        if similarity in i[0]:
            similarityfunction = i[1]
        else:
            similarityfunction = temp[0][1]

    res = similarityfunction(fp1, fp2)
    return round(res, 3)