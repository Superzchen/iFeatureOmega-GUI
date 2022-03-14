from rdkit import Chem
from .AtomProperty import GetRelativeAtomicProperty

import numpy


class CalcMoreauBroto:
    def __init__(self, lag=1, tag='m'):
        self.lag = lag
        self.tag = tag

    def __call__(self, mol, **kwargs):
        n_atom = mol.GetNumAtoms()
        GetDistanceMatrix = Chem.GetDistanceMatrix(mol)
        res = 0.0
        for i in range(n_atom):
            for j in range(n_atom):
                if GetDistanceMatrix[i, j] == self.lag:
                    atom1 = mol.GetAtomWithIdx(i)
                    atom2 = mol.GetAtomWithIdx(j)
                    temp1 = GetRelativeAtomicProperty(element=atom1.GetSymbol(), propertyname=self.tag)
                    temp2 = GetRelativeAtomicProperty(element=atom2.GetSymbol(), propertyname=self.tag)
                    res += temp1 * temp2
                else:
                    res = res + 0.0
        return numpy.log(res / 2 + 1)


class CalcMorean:
    def __init__(self, lag=1, tag='m'):
        self.lag = lag
        self.tag = tag

    def __call__(self, mol, **kwargs):
        Natom = mol.GetNumAtoms()

        prolist = []
        for i in mol.GetAtoms():
            temp = GetRelativeAtomicProperty(i.GetSymbol(), propertyname=self.tag)
            prolist.append(temp)

        aveweight = sum(prolist) / Natom

        tempp = [numpy.square(x - aveweight) for x in prolist]

        GetDistanceMatrix = Chem.GetDistanceMatrix(mol)
        res = 0.0
        index = 0
        for i in range(Natom):
            for j in range(Natom):
                if GetDistanceMatrix[i, j] == self.lag:
                    atom1 = mol.GetAtomWithIdx(i)
                    atom2 = mol.GetAtomWithIdx(j)
                    temp1 = GetRelativeAtomicProperty(element=atom1.GetSymbol(), propertyname=self.tag)
                    temp2 = GetRelativeAtomicProperty(element=atom2.GetSymbol(), propertyname=self.tag)
                    res += (temp1 - aveweight) * (temp2 - aveweight)
                    index += 1
                else:
                    res = res + 0.0

        if sum(tempp) == 0 or index == 0:
            result = 0
        else:
            result = (res / index) / (sum(tempp) / Natom)

        return result


class CalcGerary:
    def __init__(self, lag=1, tag='m'):
        self.lag = lag
        self.tag = tag

    def __call__(self, mol, **kwargs):
        Natom = mol.GetNumAtoms()

        prolist = []
        for i in mol.GetAtoms():
            temp = GetRelativeAtomicProperty(i.GetSymbol(), propertyname=self.tag)
            prolist.append(temp)

        aveweight = sum(prolist) / Natom

        tempp = [numpy.square(x - aveweight) for x in prolist]

        GetDistanceMatrix = Chem.GetDistanceMatrix(mol)
        res = 0.0
        index = 0
        for i in range(Natom):
            for j in range(Natom):
                if GetDistanceMatrix[i, j] == self.lag:
                    atom1 = mol.GetAtomWithIdx(i)
                    atom2 = mol.GetAtomWithIdx(j)
                    temp1 = GetRelativeAtomicProperty(element=atom1.GetSymbol(), propertyname=self.tag)
                    temp2 = GetRelativeAtomicProperty(element=atom2.GetSymbol(), propertyname=self.tag)
                    res = res + numpy.square(temp1 - temp2)
                    index = index + 1
                else:
                    res = res + 0.0

        if sum(tempp) == 0 or index == 0:
            result = 0
        else:
            result = (res / index / 2) / (sum(tempp) / (Natom - 1))
        return result