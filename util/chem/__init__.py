from .autocor import *
from .charge import *
from .connectivity import *
from .constitution import *
from .estate import *
from .fingerprint import *
from .topology import *
from .kappa import *
from .property import *
from .basak import *
from rdkit.Chem import Descriptors as desc
import pandas as pd

ATSm1 = CalcMoreauBroto(lag=1, tag='m')
ATSm2 = CalcMoreauBroto(lag=2, tag='m')
ATSm3 = CalcMoreauBroto(lag=3, tag='m')
ATSm4 = CalcMoreauBroto(lag=4, tag='m')
ATSm5 = CalcMoreauBroto(lag=5, tag='m')
ATSm6 = CalcMoreauBroto(lag=6, tag='m')
ATSm7 = CalcMoreauBroto(lag=7, tag='m')
ATSm8 = CalcMoreauBroto(lag=8, tag='m')
ATSv1 = CalcMoreauBroto(lag=1, tag='V')
ATSv2 = CalcMoreauBroto(lag=2, tag='V')
ATSv3 = CalcMoreauBroto(lag=3, tag='V')
ATSv4 = CalcMoreauBroto(lag=4, tag='V')
ATSv5 = CalcMoreauBroto(lag=5, tag='V')
ATSv6 = CalcMoreauBroto(lag=6, tag='V')
ATSv7 = CalcMoreauBroto(lag=7, tag='V')
ATSv8 = CalcMoreauBroto(lag=8, tag='V')
ATSe1 = CalcMoreauBroto(lag=1, tag='En')
ATSe2 = CalcMoreauBroto(lag=2, tag='En')
ATSe3 = CalcMoreauBroto(lag=3, tag='En')
ATSe4 = CalcMoreauBroto(lag=4, tag='En')
ATSe5 = CalcMoreauBroto(lag=5, tag='En')
ATSe6 = CalcMoreauBroto(lag=6, tag='En')
ATSe7 = CalcMoreauBroto(lag=7, tag='En')
ATSe8 = CalcMoreauBroto(lag=8, tag='En')
ATSp1 = CalcMoreauBroto(lag=1, tag='alpha')
ATSp2 = CalcMoreauBroto(lag=2, tag='alpha')
ATSp3 = CalcMoreauBroto(lag=3, tag='alpha')
ATSp4 = CalcMoreauBroto(lag=4, tag='alpha')
ATSp5 = CalcMoreauBroto(lag=5, tag='alpha')
ATSp6 = CalcMoreauBroto(lag=6, tag='alpha')
ATSp7 = CalcMoreauBroto(lag=7, tag='alpha')
ATSp8 = CalcMoreauBroto(lag=8, tag='alpha')

MATSm1 = CalcMorean(lag=1, tag='m')
MATSm2 = CalcMorean(lag=2, tag='m')
MATSm3 = CalcMorean(lag=3, tag='m')
MATSm4 = CalcMorean(lag=4, tag='m')
MATSm5 = CalcMorean(lag=5, tag='m')
MATSm6 = CalcMorean(lag=6, tag='m')
MATSm7 = CalcMorean(lag=7, tag='m')
MATSm8 = CalcMorean(lag=8, tag='m')
MATSv1 = CalcMorean(lag=1, tag='V')
MATSv2 = CalcMorean(lag=2, tag='V')
MATSv3 = CalcMorean(lag=3, tag='V')
MATSv4 = CalcMorean(lag=4, tag='V')
MATSv5 = CalcMorean(lag=5, tag='V')
MATSv6 = CalcMorean(lag=6, tag='V')
MATSv7 = CalcMorean(lag=7, tag='V')
MATSv8 = CalcMorean(lag=8, tag='V')
MATSe1 = CalcMorean(lag=1, tag='En')
MATSe2 = CalcMorean(lag=2, tag='En')
MATSe3 = CalcMorean(lag=3, tag='En')
MATSe4 = CalcMorean(lag=4, tag='En')
MATSe5 = CalcMorean(lag=5, tag='En')
MATSe6 = CalcMorean(lag=6, tag='En')
MATSe7 = CalcMorean(lag=7, tag='En')
MATSe8 = CalcMorean(lag=8, tag='En')
MATSp1 = CalcMorean(lag=1, tag='alpha')
MATSp2 = CalcMorean(lag=2, tag='alpha')
MATSp3 = CalcMorean(lag=3, tag='alpha')
MATSp4 = CalcMorean(lag=4, tag='alpha')
MATSp5 = CalcMorean(lag=5, tag='alpha')
MATSp6 = CalcMorean(lag=6, tag='alpha')
MATSp7 = CalcMorean(lag=7, tag='alpha')
MATSp8 = CalcMorean(lag=8, tag='alpha')

GATSm1 = CalcGerary(lag=1, tag='m')
GATSm2 = CalcGerary(lag=2, tag='m')
GATSm3 = CalcGerary(lag=3, tag='m')
GATSm4 = CalcGerary(lag=4, tag='m')
GATSm5 = CalcGerary(lag=5, tag='m')
GATSm6 = CalcGerary(lag=6, tag='m')
GATSm7 = CalcGerary(lag=7, tag='m')
GATSm8 = CalcGerary(lag=8, tag='m')
GATSv1 = CalcGerary(lag=1, tag='V')
GATSv2 = CalcGerary(lag=2, tag='V')
GATSv3 = CalcGerary(lag=3, tag='V')
GATSv4 = CalcGerary(lag=4, tag='V')
GATSv5 = CalcGerary(lag=5, tag='V')
GATSv6 = CalcGerary(lag=6, tag='V')
GATSv7 = CalcGerary(lag=7, tag='V')
GATSv8 = CalcGerary(lag=8, tag='V')
GATSe1 = CalcGerary(lag=1, tag='En')
GATSe2 = CalcGerary(lag=2, tag='En')
GATSe3 = CalcGerary(lag=3, tag='En')
GATSe4 = CalcGerary(lag=4, tag='En')
GATSe5 = CalcGerary(lag=5, tag='En')
GATSe6 = CalcGerary(lag=6, tag='En')
GATSe7 = CalcGerary(lag=7, tag='En')
GATSe8 = CalcGerary(lag=8, tag='En')
GATSp1 = CalcGerary(lag=1, tag='alpha')
GATSp2 = CalcGerary(lag=2, tag='alpha')
GATSp3 = CalcGerary(lag=3, tag='alpha')
GATSp4 = CalcGerary(lag=4, tag='alpha')
GATSp5 = CalcGerary(lag=5, tag='alpha')
GATSp6 = CalcGerary(lag=6, tag='alpha')
GATSp7 = CalcGerary(lag=7, tag='alpha')
GATSp8 = CalcGerary(lag=8, tag='alpha')

SPP = CalcSubmolPolarityPara
LDI = CalcLocalDipoleIndex
Rnc = CalcElementCharge(0, reln_sum)
Rpc = CalcElementCharge(0, relp_sum)
Mac = CalcElementCharge(0, abs_mean)
Tac = CalcElementCharge(0, abs_sum)
Mnc = CalcElementCharge(0, n_mean)
Tnc = CalcElementCharge(0, n_sum)
Mpc = CalcElementCharge(0, p_mean)
Tpc = CalcElementCharge(0, p_sum)
Qass = CalcElementCharge(0, sqsum)
QOss = CalcElementCharge(6, sqsum)
QNss = CalcElementCharge(7, sqsum)
QCss = CalcElementCharge(8, sqsum)
QHss = CalcElementCharge(1, sqsum)
Qmin = CalcElementCharge(0, np.min)
QOmin = CalcElementCharge(6, np.min)
QNmin = CalcElementCharge(7, np.min)
QCmin = CalcElementCharge(8, np.min)
QHmin = CalcElementCharge(1, np.min)
Qmax = CalcElementCharge(0, np.max)
QOmax = CalcElementCharge(6, np.max)
QNmax = CalcElementCharge(7, np.max)
QCmax = CalcElementCharge(8, np.max)
QHmax = CalcElementCharge(1, np.max)

mChi1 = MeanRandic
Chi0 = Chinp(n_path=0)
Chi1 = Chinp(n_path=0)
Chi2 = Chinp(n_path=2)
Chi3 = Chinp(n_path=3)
Chi4 = Chinp(n_path=4)
Chi5 = Chinp(n_path=5)
Chi6 = Chinp(n_path=6)
Chi7 = Chinp(n_path=7)
Chi8 = Chinp(n_path=8)
Chi9 = Chinp(n_path=9)
Chi10 = Chinp(n_path=10)
Chi3c = Chinc(tag='3', is_hk=False)
Chi4c = Chinc(tag='4', is_hk=False)
Chi4pc = Chinc(tag='4p', is_hk=False)
Chi3ch = Chinch(n_cycle=3)
Chi4ch = Chinch(n_cycle=4)
Chi5ch = Chinch(n_cycle=5)
Chi6ch = Chinch(n_cycle=6)
Chiv0 = Chivnp(n_path=0)
Chiv1 = Chivnp(n_path=1)
Chiv2 = Chivnp(n_path=2)
Chiv3 = Chivnp(n_path=3)
Chiv4 = Chivnp(n_path=4)
Chiv5 = Chivnp(n_path=5)
Chiv6 = Chivnp(n_path=6)
Chiv7 = Chivnp(n_path=7)
Chiv8 = Chivnp(n_path=8)
Chiv9 = Chivnp(n_path=9)
Chiv10 = Chivnp(n_path=10)
dchi0 = DeltaChi(Chivnp(0), Chinp(0))
dchi1 = DeltaChi(Chivnp(1), Chinp(1))
dchi2 = DeltaChi(Chivnp(2), Chinp(2))
dchi3 = DeltaChi(Chivnp(3), Chinp(3))
dchi4 = DeltaChi(Chivnp(4), Chinp(4))
Chiv3c = Chinc(tag='3', is_hk=True)
Chiv4c = Chinc(tag='4', is_hk=True)
Chiv4pc = Chinc(tag='4p', is_hk=True)
Chiv3ch = Chivnch(n_cycle=3)
Chiv4ch = Chivnch(n_cycle=4)
Chiv5ch = Chivnch(n_cycle=5)
Chiv6ch = Chivnch(n_cycle=6)
knotpv = DeltaChi(Chinc(tag='3', is_hk=True), Chinc(tag='4p', is_hk=True))
knotp = DeltaChi(Chinc(tag='3', is_hk=False), Chinc(tag='4p', is_hk=False))

Weight = desc.MolWt
nhyd = FragCounter('[H]')
nhal = FragCounter('[F,Cl,Br,I]')
nhet = AllChem.CalcNumHeteroatoms
nring = AllChem.CalcNumRings
nrot = AllChem.CalcNumRotatableBonds
ndonr = AllChem.CalcNumHBD
naccr = AllChem.CalcNumHBA

nhev = FragCounter('[!H]')
ncof = FragCounter('F')
ncocl = FragCounter('Cl')
ncobr = FragCounter('Br')
ncoi = FragCounter('I')
ncarb = FragCounter('C')
nphos = FragCounter('P')
nsulph = FragCounter('S')
noxy = FragCounter('O')
nnitro = FragCounter('N')
nsb = FragCounter('[*]-[*]')
ndb = FragCounter('[*]=[*]')
ntb = FragCounter('[*]#[*]')
naro = FragCounter('[*]:[*]')
nta = TotalAtom()
AWeight = TotalAtom(is_weight=True)
PC1 = PathsOfLengthN(1)
PC2 = PathsOfLengthN(2)
PC3 = PathsOfLengthN(3)
PC4 = PathsOfLengthN(4)
PC5 = PathsOfLengthN(5)
PC6 = PathsOfLengthN(6)

value = CalcEstateValue
max = CalcMaxAtomTypeEState
min = CalcMinAtomTypeEState
Shev = CalcHeavyAtomEState
Scar = CalcCAtomEState
Shal = CalculateHalogenEState
Shet = CalculateHeteroEState
Save = CalcAverageEState
Smax = CalcMaxEState
Smin = CalcMinEState
DS = CalcDiffMaxMinEState

slogPVSA = MolSurf.SlogP_VSA_
MRVSA = MolSurf.SMR_VSA_
PEOEVSA = MolSurf.PEOE_VSA_
EstateVSA = EVSA.EState_VSA_
VSAEstate = EVSA.VSA_EState_

topological = CalcDaylightFingerprint
Estate = CalcEstateFingerprint
atompairs = CalcAtomPairsFingerprint
torsions = CalculateTopologicalTorsionFingerprint
# morgan = CalculateMorganFingerprint
ECFP4 = CalcMorganFingerprint(radius=2, useFeatures=False)
ECFP6 = CalcMorganFingerprint(radius=3, useFeatures=False)
FCFP4 = CalcMorganFingerprint(radius=2, useFeatures=True)
FCFP6 = CalcMorganFingerprint(radius=3, useFeatures=True)
MACCS = CalculateMACCSFingerprint

W = WienerIdx()
AW = WienerIdx(is_average=True)
J = graph.BalabanJ
Thara = NumHarary
Tsch = SchiultzIdx
Tigdi = GraphDistIdx
Platt = CalcPlatt
Xu = XuIdx
Pol = NumPolarity
Dz = PoglianiIdx
Ipc = IpcIdx
BertzCT = BertzCT
GMTI = GutmanTopo
ZM1 = Zagreb1
ZM2 = Zagreb2
MZM1 = MZagreb1
MZM2 = MZagreb2
Qindex = Quadratic
diametert = Diameter
radiust = Radius
petitjeant = Petitjean
Sito = SimpleTopovIdx
Hato = HarmonicTopovIdx
Geto = GeometricTopovIdx
Arto = ArithmeticTopoIdx

kappa1 = CalcKappa(is_alpha=False, n_bond=1)
kappa2 = CalcKappa(is_alpha=False, n_bond=2)
kappa3 = CalcKappa(is_alpha=False, n_bond=3)
kappam1 = CalcKappa(is_alpha=True, n_bond=1)
kappam2 = CalcKappa(is_alpha=True, n_bond=2)
kappam3 = CalcKappa(is_alpha=True, n_bond=3)
phi = Flexibility

LogP = Crippen.MolLogP
MR = Crippen.MolMR
LabuteASA = MS.pyLabuteASA
TPSA = MS.TPSA
Hy = CalculateHydrophilicityFactor
UI = CalculateUnsaturationIndex

# Basak
CIC0 = CalcBasakCIC0
CIC1 = CalcBasakCICn(num_path=2)
CIC2 = CalcBasakCICn(num_path=3)
CIC3 = CalcBasakCICn(num_path=4)
CIC4 = CalcBasakCICn(num_path=5)
CIC5 = CalcBasakCICn(num_path=6)
CIC6 = CalcBasakCICn(num_path=7)
SIC0 = CalcBasakSIC0
SIC1 = CalcBasakSICn(num_path=2)
SIC2 = CalcBasakSICn(num_path=3)
SIC3 = CalcBasakSICn(num_path=4)
SIC4 = CalcBasakSICn(num_path=5)
SIC5 = CalcBasakSICn(num_path=6)
SIC6 = CalcBasakSICn(num_path=7)
IC0 = CalcBasakIC0
IC1 = CalcBasakICn(num_path=2)
IC2 = CalcBasakICn(num_path=3)
IC3 = CalcBasakICn(num_path=4)
IC4 = CalcBasakICn(num_path=5)
IC5 = CalcBasakICn(num_path=6)
IC6 = CalcBasakICn(num_path=7)

# Burden
from .burden import *
bcutp = CalcBurden(label='alpha')
bcute = CalcBurden(label='En')
bcutv = CalcBurden(label='V')
bcutm = CalcBurden(label='m')

# Pharmacophore
from .cats import *
