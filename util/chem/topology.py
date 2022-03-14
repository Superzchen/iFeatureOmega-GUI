from rdkit import Chem
from rdkit.Chem import GraphDescriptors as graph
from rdkit.Chem.EState import GetPrincipleQuantumNumber
import numpy as np


class WienerIdx:
    def __init__(self, is_average=False):
        self.is_average = is_average

    def __call__(self, m, **kwargs):
       amat = Chem.GetDistanceMatrix(m)
       res = sum(sum(amat))
       N = m.GetNumAtoms()
       if self.is_average:
           res = res / N / (N-1)
       return res / 2


def NumHarary(mol, **kwargs):
    dist = np.array(Chem.GetDistanceMatrix(mol), 'd')
    return 1.0 / 2 * (sum(1.0 / dist[dist != 0]))


def SchiultzIdx(mol, **kwargs):
    Distance = np.array(Chem.GetDistanceMatrix(mol), 'd')
    Adjacent = np.array(Chem.GetAdjacencyMatrix(mol), 'd')
    VertexDegree = sum(Adjacent)
    return sum(np.dot((Distance + Adjacent), VertexDegree))


def GraphDistIdx(mol, **kwargs):
    Distance = Chem.GetDistanceMatrix(mol)
    n = int(Distance.max())
    res = 0.0
    for i in range(n):
        # print Distance==i+1
        temp = 1. / 2 * sum(sum(Distance == i + 1))
        # print temp
        res = res + temp ** 2
    return np.log10(res)


def CalcPlatt(mol, **kwargs):
    cc = [x.GetBeginAtom().GetDegree() + x.GetEndAtom().GetDegree() - 2 for x in mol.GetBonds()]
    return sum(cc)


def XuIdx(mol, **kwargs):
    nAT = mol.GetNumAtoms()
    deltas = [x.GetDegree() for x in mol.GetAtoms()]
    Distance = Chem.GetDistanceMatrix(mol)
    sigma = np.sum(Distance, axis=1)
    temp1 = 0.0
    temp2 = 0.0
    for i in range(nAT):
        temp1 = temp1 + deltas[i] * ((sigma[i]) ** 2)
        temp2 = temp2 + deltas[i] * (sigma[i])
    Xu = np.sqrt(nAT) * np.log(temp1 / temp2)
    return Xu


def NumPolarity(mol, **kwargs):
    dist = Chem.GetDistanceMatrix(mol)
    res = 1. / 2 * sum(sum(dist == 3))
    return res


def PoglianiIdx(mol, **kwargs):
    res = 0.0
    for atom in mol.GetAtoms():
        n = atom.GetAtomicNum()
        nV = Chem.GetPeriodicTable().GetNOuterElecs(n)
        mP = GetPrincipleQuantumNumber(n)
        res = res + (nV + 0.0) / mP
    return res


def IpcIdx(mol, **kwargs):
    return np.log10(graph.Ipc(mol))


def BertzCT(mol, **kwargs):
    return np.log10(graph.BertzCT(mol))


def GutmanTopo(mol, **kwargs):
    nAT = mol.GetNumAtoms()
    deltas = [x.GetDegree() for x in mol.GetAtoms()]
    dist = Chem.GetDistanceMatrix(mol)
    res = 0.0
    for i in range(nAT):
        for j in range(i + 1, nAT):
            res = res + deltas[i] * deltas[j] * dist[i, j]

    return np.log10(res)


def Zagreb1(mol, **kwargs):
    deltas = [x.GetDegree() for x in mol.GetAtoms()]
    return sum(np.array(deltas) ** 2)


def Zagreb2(mol, **kwargs):
    ke = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
    return sum(ke)


def MZagreb1(mol, **kwargs):
    deltas = [x.GetDegree() for x in mol.GetAtoms()]
    while 0 in deltas:
        deltas.remove(0)
    deltas = np.array(deltas, 'd')
    res = sum((1. / deltas) ** 2)
    return res


def MZagreb2(mol, **kwargs):
    cc = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
    while 0 in cc:
        cc.remove(0)
    cc = np.array(cc, 'd')
    res = sum((1. / cc) ** 2)
    return res


def Quadratic(mol, **kwargs):
    M = Zagreb1(mol)
    N = mol.GetNumAtoms()
    return 3 - 2 * N + M / 2.0


def Diameter(mol, **kwargs):
    dist = Chem.GetDistanceMatrix(mol)
    return dist.max()


def Radius(mol, **kwargs):
    Distance = Chem.GetDistanceMatrix(mol)
    temp = []
    for i in Distance:
        temp.append(max(i))
    return min(temp)


def Petitjean(mol, **kwargs):
    diameter =Diameter(mol)
    radius = Radius(mol)
    return 1 - radius / float(diameter)


def SimpleTopovIdx(mol, **kwargs):
    deltas = graph._hkDeltas(mol, skipHs=0)
    while 0 in deltas:
        deltas.remove(0)
    deltas = np.array(deltas, 'd')

    res = np.prod(deltas)

    return np.log(res)


def HarmonicTopovIdx(mol, **kwargs):
    deltas = graph._hkDeltas(mol, skipHs=0)
    while 0 in deltas:
        deltas.remove(0)
    deltas = np.array(deltas, 'd')
    nAtoms = mol.GetNumAtoms()

    res = nAtoms / sum(1. / deltas)

    return res


def GeometricTopovIdx(mol, **kwargs):
    nAtoms = mol.GetNumAtoms()
    deltas = graph._hkDeltas(mol, skipHs=0)
    while 0 in deltas:
        deltas.remove(0)
    deltas = np.array(deltas, 'd')

    temp = np.prod(deltas)
    res = np.power(temp, 1. / nAtoms)

    return res


def ArithmeticTopoIdx(mol, **kwargs):
    nAtoms = mol.GetNumAtoms()
    nBonds = mol.GetNumBonds()
    res = 2. * nBonds / nAtoms
    return res
