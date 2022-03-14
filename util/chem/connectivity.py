from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
from rdkit.Chem import GraphDescriptors as graph


class Chinp:
    """
    #################################################################
    Calculation of molecular connectivity chi index for path order n
    #################################################################
    """
    def __init__(self, n_path=2):
        self.n_path = n_path

    def __call__(self, mol, **kwargs):
        accum = 0.0
        if self.n_path in [0, 1]:
            if self.n_path == 0:
                deltas = [x.GetDegree() for x in mol.GetAtoms()]
            else:
                deltas = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
            while 0 in deltas:
                deltas.remove(0)
            deltas = np.array(deltas, 'd')
            accum = sum(np.sqrt(1. / deltas))
        else:
            deltas = [x.GetDegree() for x in mol.GetAtoms()]
            for path in Chem.FindAllPathsOfLengthN(mol, self.n_path + 1, useBonds=0):
                cAccum = 1.0
                for idx in path:
                    cAccum *= deltas[idx]
                if cAccum:
                    accum += 1. / np.sqrt(cAccum)
        return accum


class Chinch:
    """
    #################################################################
    Calculation of molecular connectivity chi index for cycles of n
    #################################################################
    """
    def __init__(self, n_cycle=3):
        self.n_cycle = n_cycle

    def __call__(self, mol, **kwargs):
        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        for tup in mol.GetRingInfo().AtomRings():
            cAccum = 1.0
            if len(tup) == self.n_cycle:
                for idx in tup:
                    cAccum *= deltas[idx]
                if cAccum:
                    accum += 1. / np.sqrt(cAccum)

        return accum


def MeanRandic(mol):
    """
    #################################################################
    Calculation of mean chi1 (Randic) connectivity index.

    ---->mchi1

    Usage:

        result=MeanRandic(mol)

        Input: mol is a molecule object.

        Output: result is a numeric value
    #################################################################
    """
    cc = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
    while 0 in cc:
        cc.remove(0)
    cc = np.array(cc, 'd')
    res = np.mean(np.sqrt(1. / cc))

    return res


class Chinc:
    def __init__(self, tag='3', is_hk=False):
        patts = {'3': '*~*(~*)~*',
                 '4': '*~*(~*)(~*)~*',
                 '4p': '*~*(~*)~*~*',
                 }
        self.patt = Chem.MolFromSmarts(patts[tag])
        self.is_hk = is_hk

    def __call__(self, mol, **kwargs):
        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        HPatt = mol.GetSubstructMatches(self.patt)
        for cluster in HPatt:
            if self.is_hk:
                deltas = [_AtomHKDeltas(mol.GetAtomWithIdx(x)) for x in cluster]
            else:
                deltas = [mol.GetAtomWithIdx(x).GetDegree() for x in cluster]
            while 0 in deltas:
                deltas.remove(0)
            if deltas != []:
                deltas1 = np.array(deltas, np.float)
                accum = accum + 1. / np.sqrt(deltas1.prod())
        return accum


class Chivnp:
    """#################################################################
    Calculation of valence molecular connectivity chi index for path order n
    #################################################################
    """
    def __init__(self, n_path):
        self.n_path = n_path

    def __call__(self, mol, **kwargs):
        deltas = graph._hkDeltas(mol, skipHs=0)
        if self.n_path == 0:
            while 0 in deltas:
                deltas.remove(0)
            deltas = np.array(deltas, 'd')
            accum = sum(np.sqrt(1. / deltas))
        else:
            accum = 0.0
            for path in Chem.FindAllPathsOfLengthN(mol, self.n_path + 1, useBonds=0):
                cAccum = 1.0
                for idx in path:
                    cAccum *= deltas[idx]
                if cAccum:
                    accum += 1. / np.sqrt(cAccum)
        return accum


class Chivnch:
    """
    #################################################################
    Calculation of valence molecular connectivity chi index for cycles of n
    #################################################################
    """
    def __init__(self, n_cycle=3):
        self.n_cycle = n_cycle

    def __call__(self, mol, **kwargs):
        accum = 0.0
        deltas = graph._hkDeltas(mol, skipHs=0)
        for tup in mol.GetRingInfo().AtomRings():
            cAccum = 1.0
            if len(tup) == self.n_cycle:
                for idx in tup:
                    cAccum *= deltas[idx]
                if cAccum:
                    accum += 1. / np.sqrt(cAccum)

        return accum


class DeltaChi:
    def __init__(self, fn1, fn2):
        self.fn1 = fn1
        self.fn2 = fn2

    def __call__(self, mol, **kwargs):
        return abs(self.fn1(mol) - self.fn2(mol))


def _AtomHKDeltas(atom, skipHs=0):
    """
    #################################################################
    Calculation of modified delta value for a molecule
    #################################################################
    """
    res = []
    n = atom.GetAtomicNum()
    if n > 1:
        nV = Chem.GetPeriodicTable().GetNOuterElecs(n)
        nHs = atom.GetTotalNumHs()
        if n < 10:
            res.append(float(nV - nHs))
        else:
            res.append(float(nV - nHs) / float(n - nV - 1))
    elif not skipHs:
        res.append(0.0)
    return res

