# -*- coding: utf-8 -*-
#  Copyright (c) 2016-2017, Zhijiang Yao, Jie Dong and Dongsheng Cao
#  All rights reserved.
#  This file is part of the PyBioMed.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the PyBioMed source tree.
"""
##############################################################################
The calculation of some commonly used basak information index  based on its
topological structure. You can get 21 molecular connectivity descriptors.
You can freely use and distribute it. If you hava  any problem, you could
contact with us timely!
Authors: Zhijiang Yao and Dongsheng Cao.
Date: 2016.06.04
Email: gadsby@163.
##############################################################################
"""


# Core Library modules
import copy

# Third party modules
import numpy
from rdkit import Chem

Version = 1.0

############################################################################


def _CalcEntropy(Probability):
    """
    #################################################################
    **Internal used only**
    Calculation of entropy (Information content) for probability given
    #################################################################
    """
    res = 0.0
    for i in Probability:
        if i != 0:
            res = res - i * numpy.log2(i)

    return res


def CalcBasakIC0(mol, **kwargs):

    """
    #################################################################
    Obtain the information content with order 0 proposed by Basak
    ---->IC0
    #################################################################
    """

    BasakIC = 0.0
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = []
    for i in range(nAtoms):
        at = Hmol.GetAtomWithIdx(i)
        IC.append(at.GetAtomicNum())
    Unique = numpy.unique(IC)
    NAtomType = len(Unique)
    NTAtomType = numpy.zeros(NAtomType, numpy.float)
    for i in range(NAtomType):
        NTAtomType[i] = IC.count(Unique[i])

    if nAtoms != 0:
        # print sum(NTAtomType/nAtoms)
        BasakIC = _CalcEntropy(NTAtomType / nAtoms)
    else:
        BasakIC = 0.0

    return BasakIC


def CalcBasakSIC0(mol, **kwargs):
    """
    #################################################################
    Obtain the structural information content with order 0
    proposed by Basak
    ---->SIC0
    #################################################################
    """

    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalcBasakIC0(mol)
    if nAtoms <= 1:
        BasakSIC = 0.0
    else:
        BasakSIC = IC / numpy.log2(nAtoms)

    return BasakSIC


def CalcBasakCIC0(mol, **kwargs):

    """
    #################################################################
    Obtain the complementary information content with order 0
    proposed by Basak
    ---->CIC0
    #################################################################
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalcBasakIC0(mol)
    if nAtoms <= 1:
        BasakCIC = 0.0
    else:
        BasakCIC = numpy.log2(nAtoms) - IC

    return BasakCIC


class CalcBasakICn:
    def __init__(self, num_path=1):
        self.num_path = num_path

    def __call__(self, mol, **kwargs):
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        TotalPath = Chem.FindAllPathsOfLengthN(Hmol, self.num_path, useBonds=0, useHs=1)
        if len(TotalPath) == 0:
            BasakIC = 0.0
        else:
            IC = {}
            for i in range(nAtoms):
                temp = []
                at = Hmol.GetAtomWithIdx(i)
                temp.append([at.GetAtomicNum()])
                for index in TotalPath:
                    if i == index[0]:
                        temp.append(
                            [Hmol.GetAtomWithIdx(kk).GetAtomicNum() for kk in index[1:]]
                        )
                    if i == index[-1]:
                        cds = list(index)
                        cds.reverse()
                        temp.append(
                            [Hmol.GetAtomWithIdx(kk).GetAtomicNum() for kk in cds[1:]]
                        )
                # print temp

                IC[str(i)] = temp
            cds = []
            for value in IC.values():
                value.sort()
                cds.append(value)
            kkk = list(range(len(cds)))
            aaa = copy.deepcopy(kkk)
            res = []
            for i in aaa:
                if i in kkk:
                    jishu = 0
                    kong = []
                    temp1 = cds[i]
                    for j in aaa:
                        if cds[j] == temp1:
                            jishu = jishu + 1
                            kong.append(j)
                    for ks in kong:
                        kkk.remove(ks)
                    res.append(jishu)

            # print res
            BasakIC = _CalcEntropy(numpy.array(res, numpy.float) / sum(res))

        return BasakIC


class CalcBasakSICn:
    def __init__(self, num_path=1):
        self.num_path = num_path

    def __call__(self, mol, **kwargs):
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = CalcBasakICn(num_path=self.num_path)(mol)
        if nAtoms <= 1:
            BasakSIC = 0.0
        else:
            BasakSIC = IC / numpy.log2(nAtoms)

        return BasakSIC


class CalcBasakCICn:
    def __init__(self, num_path=1):
        self.num_path = num_path

    def __call__(self, mol, **kwargs):
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = CalcBasakICn(num_path=self.num_path)(mol)
        if nAtoms <= 1:
            BasakCIC = 0.0
        else:
            BasakCIC = numpy.log2(nAtoms) - IC

        return BasakCIC

