import numpy
import numpy.linalg
from rdkit import Chem
from .AtomProperty import GetRelativeAtomicProperty


def _GetBurdenMatrix(mol, propertylabel="m"):
    mol = Chem.AddHs(mol)
    Natom = mol.GetNumAtoms()

    AdMatrix = Chem.GetAdjacencyMatrix(mol)
    bondindex = numpy.argwhere(AdMatrix)
    AdMatrix1 = numpy.array(AdMatrix, dtype=numpy.float32)

    # The diagonal elements of B, Bii, are either given by
    # the carbon normalized atomic mass,
    # van der Waals volume, Sanderson electronegativity,
    # and polarizability of atom i.

    for i in range(Natom):
        atom = mol.GetAtomWithIdx(i)
        temp = GetRelativeAtomicProperty(
            element=atom.GetSymbol(), propertyname=propertylabel
        )
        AdMatrix1[i, i] = round(temp, 3)

    # The element of B connecting atoms i and j, Bij,
    # is equal to the square root of the bond
    # order between atoms i and j.

    for i in bondindex:
        bond = mol.GetBondBetweenAtoms(int(i[0]), int(i[1]))
        if bond.GetBondType().name == "SINGLE":
            AdMatrix1[i[0], i[1]] = round(numpy.sqrt(1), 3)
        if bond.GetBondType().name == "DOUBLE":
            AdMatrix1[i[0], i[1]] = round(numpy.sqrt(2), 3)
        if bond.GetBondType().name == "TRIPLE":
            AdMatrix1[i[0], i[1]] = round(numpy.sqrt(3), 3)
        if bond.GetBondType().name == "AROMATIC":
            AdMatrix1[i[0], i[1]] = round(numpy.sqrt(1.5), 3)

    ##All other elements of B (corresponding non bonded
    # atom pairs) are set to 0.001
    bondnonindex = numpy.argwhere(AdMatrix == 0)

    for i in bondnonindex:
        if i[0] != i[1]:

            AdMatrix1[i[0], i[1]] = 0.001

    return numpy.real(numpy.linalg.eigvals(AdMatrix1))


class CalcBurden:
    def __init__(self, label='m'):
        self.label = label

    def __call__(self, mol, **kwargs):
        temp = _GetBurdenMatrix(mol, propertylabel=self.label)
        temp1 = numpy.sort(temp[temp >= 0])
        temp2 = numpy.sort(numpy.abs(temp[temp < 0]))

        if len(temp1) < 8:
            temp1 = numpy.concatenate((numpy.zeros(8), temp1))
        if len(temp2) < 8:
            temp2 = numpy.concatenate((numpy.zeros(8), temp2))
        bcutvalue = numpy.concatenate((temp2[-8:], temp1[-8:]))
        return bcutvalue




