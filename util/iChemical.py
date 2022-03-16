import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pPath, 'chem'))
import pandas as pd
from rdkit import Chem
import numpy as np
from chem import *

class Ligand:
    def __init__(self, fps=[]):
        self.fps = fps

    def __call__(self, mols, **kwargs):
        df = pd.DataFrame()
        for i, mol in enumerate(mols):
            for fp in self.fps:
                coder = eval(fp)
                code = coder(mol, **kwargs)
                if type(code) == list or type(code) == np.ndarray:
                    for j, c in enumerate(code):
                        df.loc[i, fp+str(j)] = c
                else:
                    df.loc[i, fp] = code                    
        return df


if __name__ == '__main__':
    # mol = Chem.MolFromSmiles('Clc1ccccc1Br')
    mol = Chem.MolFromSmiles('C(C=CC1)=C(C=1C(=O)O)O')    

    autocor = ['ATSm1', 'ATSm2', 'ATSm3', 'ATSm4', 'ATSm5', 'ATSm6', 'ATSm7', 'ATSm8',
               'ATSv1', 'ATSv2', 'ATSv3', 'ATSv4', 'ATSv5', 'ATSv6', 'ATSv7', 'ATSv8',
               'ATSe1', 'ATSe2', 'ATSe3', 'ATSe4', 'ATSe5', 'ATSe6', 'ATSe7', 'ATSe8',
               'ATSp1', 'ATSp2', 'ATSp3', 'ATSp4', 'ATSp5', 'ATSp6', 'ATSp7', 'ATSp8',
               'MATSm1', 'MATSm2', 'MATSm3', 'MATSm4', 'MATSm5', 'MATSm6', 'MATSm7', 'MATSm8',
               'MATSv1', 'MATSv2', 'MATSv3', 'MATSv4', 'MATSv5', 'MATSv6', 'MATSv7', 'MATSv8',
               'MATSe1', 'MATSe2', 'MATSe3', 'MATSe4', 'MATSe5', 'MATSe6', 'MATSe7', 'MATSe8',
               'MATSp1', 'MATSp2', 'MATSp3', 'MATSp4', 'MATSp5', 'MATSp6', 'MATSp7', 'MATSp8',
               'GATSm1', 'GATSm2', 'GATSm3', 'GATSm4', 'GATSm5', 'GATSm6', 'GATSm7', 'GATSm8',
               'GATSv1', 'GATSv2', 'GATSv3', 'GATSv4', 'GATSv5', 'GATSv6', 'GATSv7', 'GATSv8',
               'GATSe1', 'GATSe2', 'GATSe3', 'GATSe4', 'GATSe5', 'GATSe6', 'GATSe7', 'GATSe8',
               'GATSp1', 'GATSp2', 'GATSp3', 'GATSp4', 'GATSp5', 'GATSp6', 'GATSp7', 'GATSp8']

    charge = ['SPP', 'LDI', 'Rnc', 'Rpc', 'Mac', 'Tac', 'Mnc', 'Tnc', 'Mpc', 'Tpc', 'Qass', 'QOss', 'QNss', 'QCss',
              'QHss', 'Qmin', 'QOmin', 'QNmin', 'QCmin', 'QHmin', 'Qmax', 'QOmax', 'QNmax', 'QCmax', 'QHmax']

    connectivity = ['Chi0', 'Chi1', 'mChi1', 'Chi2', 'Chi3', 'Chi4', 'Chi5', 'Chi6', 'Chi7', 'Chi8', 'Chi9', 'Chi10',
                    'Chi3c', 'Chi4c', 'Chi4pc', 'Chi3ch', 'Chi4ch', 'Chi5ch', 'Chi6ch', 'Chiv0', 'Chiv1', 'Chiv2',
                    'Chiv3', 'Chiv4', 'Chiv5', 'Chiv6', 'Chiv7', 'Chiv8', 'Chiv9', 'Chiv10', 'dchi0', 'dchi1', 'dchi2',
                    'dchi3', 'dchi4', 'Chiv3c', 'Chiv4c', 'Chiv4pc', 'Chiv3ch', 'Chiv4ch', 'Chiv5ch', 'Chiv6ch',
                    'knotpv', 'knotp']

    constitue = ['nhyd', 'nhal', 'nhet', 'nhev', 'ncof', 'ncocl', 'ncobr', 'ncoi', 'ncarb', 'nphos',
                 'nsulph', 'noxy', 'nnitro', 'nring', 'nrot', 'ndonr', 'naccr', 'nsb', 'ndb',
                 'ntb', 'naro', 'nta', 'AWeight', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']

    estate = ['value', 'max', 'min', 'Shev', 'Scar', 'Shal', 'Shet', 'Save', 'Smax', 'Smin', 'DS']

    fp = ['topological', 'Estate', 'atompairs', 'torsions', 'morgan', 'MACCS']

    topo = ['AW', 'J', 'Thara', 'Tsch', 'Tigdi', 'Platt', 'Xu', 'Pol', 'Dz', 'Ipc',
            'BertzCT', 'GMTI', 'ZM1', 'ZM2', 'MZM1', 'MZM2', 'Qindex', 'diametert',
            'radiust', 'petitjeant', 'Sito', 'Hato', 'Geto', 'Arto']

    kappa = ['kappa1', 'kappa2', 'kappa3', 'kappam1', 'kappam2', 'kappam3', 'phi']
    props = ['LogP', 'MR', 'TPSA', 'Hy', 'UI']
    moe = ['LabuteASA', 'TPSA', 'slogPVSA', 'MRVSA', 'PEOEVSA', 'EstateVSA', 'VSAEstate']
    fps = autocor + charge + connectivity + constitue + estate + fp + topo

    ligand = Ligand(fps=autocor)
    df = ligand([mol, mol])
    print(df)
