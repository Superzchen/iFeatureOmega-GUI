#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, sys, re

from matplotlib.pyplot import plot
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from PyQt5.QtWidgets import (QApplication, QTableWidgetItem, QWidget, QTabWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QGroupBox, QTextEdit,
                             QVBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem, QFormLayout, QTableWidget, QHeaderView, QAbstractItemView, 
                             QSplitter, QGridLayout, QMessageBox, QInputDialog)
from PyQt5.QtGui import QIcon, QFont, QMovie
from PyQt5.QtCore import Qt, pyqtSignal
from util import (InputDialog, PlotWidgets, TableWidget, iSequence, iStructure, iChemical, DataAnalysis, CheckAccPseParameter)
from rdkit import Chem
import qdarkstyle
import threading
import numpy as np
import pandas as pd
import tempfile
import time
import sip
import copy


class IFeatureOmegaGui(QTabWidget):
    # 信号定义为类属性，不能放在__init__里面
    # global signal
    display_error_signal = pyqtSignal(str)
    display_warning_signal = pyqtSignal(str)

    # protein signal
    protein_message_signal = pyqtSignal(str)
    protein_display_signal = pyqtSignal()
    
    # dna signal
    dna_message_signal = pyqtSignal(str)
    dna_display_signal = pyqtSignal()

    # rna signal
    rna_message_signal = pyqtSignal(str)
    rna_display_signal = pyqtSignal()

    # structure signal
    structure_message_signal = pyqtSignal(str)
    structure_display_signal = pyqtSignal()

    # chemical signal
    chemical_message_signal = pyqtSignal(str)
    chemical_display_signal = pyqtSignal()

    # analysis signal
    analysis_message_signal = pyqtSignal(str)
    analysis_display_signal = pyqtSignal()

    # plot signal
    plot_message_signal = pyqtSignal(str)
    plot_display_signal = pyqtSignal()

    def __init__(self):
        super(IFeatureOmegaGui, self).__init__()

        # gif
        self.gif = QMovie(os.path.join(pPath, 'images', 'progress_bar.gif'))
        self.working_dir = tempfile.mkdtemp()        

        self.display_error_signal.connect(self.display_error_msg)
        self.display_warning_signal.connect(self.display_warning_msg)

        # Protein
        self.protein_sequence_file = None
        self.protein_selected_descriptors = set([])
        self.protein_para_dict = {
            'EAAC': {'sliding_window': 5},
            'CKSAAP': {'kspace': 3},
            'EGAAC': {'sliding_window': 5},
            'CKSAAGP': {'kspace': 3},
            'AAIndex': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101'},
            'NMBroto': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3,},
            'Moran': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3,},
            'Geary': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3,},
            'KSCTriad': {'kspace': 3},
            'SOCNumber': {'nlag': 3},
            'QSOrder': {'nlag': 3, 'weight': 0.05},
            'PAAC': {'weight': 0.05, 'lambdaValue': 3},
            'APAAC': {'weight': 0.05, 'lambdaValue': 3},
            'DistancePair': {'distance': 0, 'cp': 'cp(20)',},
            'AC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'CC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'ACC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'PseKRAAC type 1': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 2': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 3A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 3B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 4': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 5': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 6A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 6B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 6C': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 7': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 8': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 9': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 10': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 11': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 12': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 13': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 14': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 15': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 16': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},            
        } 
        self.protein_default_para = {             # default parameter for descriptors
            'sliding_window': 5,
            'kspace': 3,
            'props': ['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
            'nlag': 3,
            'weight': 0.05,
            'lambdaValue': 3,
            'PseKRAAC_model': 'g-gap',
            'g-gap': 2,
            'k-tuple': 2,
            'RAAC_clust': 1,
            'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101',            
        }
        self.protein_descriptor = None        
        self.protein_message_signal.connect(self.protein_display_message)
        self.protein_display_signal.connect(self.set_protein_table_content)
        self.protein_encodings = None
        self.protein_html_dict = {
            'heatmap': None,
            'boxplot': None,
        }

        # DNA
        self.dna_sequence_file = None
        self.dna_selected_descriptors = set([])
        self.dna_para_dict = {
            'Kmer': {'kmer': 3},
            'RCKmer': {'kmer': 3},
            'Mismatch': {'kmer': 3, 'mismatch': 1},
            'Subsequence': {'kmer': 3, 'delta': 0},
            'ENAC': {'sliding_window': 5},
            'CKSNAP': {'kspace': 3},
            'DPCP': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise'},
            'DPCP type2': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise'},
            'TPCP': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
            'TPCP type2': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
            'DAC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'nlag': 3},
            'DCC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'nlag': 3},
            'DACC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'nlag': 3},
            'TAC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'TCC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'TACC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'PseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'weight': 0.05, 'lambdaValue': 3},
            'PseKNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'weight': 0.05, 'lambdaValue': 3, 'kmer': 3},
            'PCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'weight': 0.05, 'lambdaValue': 3},
            'PCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
            'SCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'weight': 0.05, 'lambdaValue': 3},
            'SCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
            'NMBroto': {'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'Moran': {'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'Geary': {'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
        } 
        self.dna_default_para = {             # default parameter for descriptors
            'kmer': 3,
            'sliding_window': 5,
            'kspace': 3,            
            'mismatch': 1,
            'delta': 0,
            'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',
            'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)',            
            'distance': 0,
            'cp': 'cp(20)',
            'nlag': 3,
            'lambdaValue': 3,
            'weight': 0.05,            
        }
        self.dna_descriptor = None        
        self.dna_message_signal.connect(self.dna_display_message)
        self.dna_display_signal.connect(self.set_dna_table_content)
        self.dna_encodings = None

        # RNA
        self.rna_sequence_file = None
        self.rna_selected_descriptors = set([])
        self.rna_para_dict = {
            'Kmer': {'kmer': 3},
            'RCKmer': {'kmer': 3},
            'Mismatch': {'kmer': 3, 'mismatch': 1},
            'Subsequence': {'kmer': 3, 'delta': 0},
            'ENAC': {'sliding_window': 5},
            'CKSNAP': {'kspace': 3},
            'DPCP': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'DPCP type2': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},            
            'DAC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'DCC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'DACC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},            
            'PseDNC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
            'PseKNC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3, 'kmer': 3},
            'PCPseDNC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},            
            'SCPseDNC': {'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},            
            'NMBroto': {'nlag': 3, 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'Moran': {'nlag': 3, 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'Geary': {'nlag': 3, 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
        } 
        self.rna_default_para = {             # default parameter for descriptors
            'sliding_window': 5,
            'kspace': 3,
            'kmer': 3,
            'mismatch': 1,
            'delta': 0,            
            'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)',
            'distance': 0,
            'cp': 'cp(20)',
            'nlag': 3,
            'lambdaValue': 3,
            'weight': 0.05,
        }
        self.rna_descriptor = None        
        self.rna_message_signal.connect(self.rna_display_message)
        self.rna_display_signal.connect(self.set_rna_table_content)
        self.rna_encodings = None

        # Structure
        self.structure_file = None
        self.structure_selected_descriptors = None
        self.structure_para_dict = {} 
        self.structure_default_para = {             # default parameter for descriptors
            'residue_shell': (3, 30, 3),
            'atom_shell': (1, 10, 1),
        }
        self.structure_descriptor = None        
        self.structure_message_signal.connect(self.structure_display_message)
        self.structure_display_signal.connect(self.set_structure_table_content)        
        self.structure_encodings = None
        self.structure_tmpdir = None

        # Chemical
        self.chemical_smiles_file = None
        self.chemical_selected_descriptors = set([])
        self.chemical_message_signal.connect(self.chemical_display_message)
        self.chemical_display_signal.connect(self.set_chemical_table_content)
        self.chemical_descriptor = None
        self.chemical_encodings = None
        self.chemical_default_parameters = {
            'Constitution': ['nhyd', 'nhal', 'nhet', 'nhev', 'ncof', 'ncocl', 'ncobr', 'ncoi', 'ncarb', 'nphos', 'nsulph', 'noxy', 'nnitro', 'nring', 'nrot', 'ndonr', 'naccr', 'nsb', 'ndb', 'ntb', 'naro', 'nta', 'AWeight', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'],
            'Topology': ['AW', 'J', 'Thara', 'Tsch', 'Tigdi', 'Platt', 'Xu', 'Pol', 'Dz', 'Ipc', 'BertzCT', 'GMTI', 'ZM1', 'ZM2', 'MZM1', 'MZM2', 'Qindex', 'diametert', 'radiust', 'petitjeant', 'Sito', 'Hato', 'Geto', 'Arto'],
            'Connectivity': ['Chi0', 'Chi1', 'mChi1', 'Chi2', 'Chi3', 'Chi4', 'Chi5', 'Chi6', 'Chi7', 'Chi8', 'Chi9', 'Chi10', 'Chi3c', 'Chi4c', 'Chi4pc', 'Chi3ch', 'Chi4ch', 'Chi5ch', 'Chi6ch', 'Chiv0', 'Chiv1', 'Chiv2', 'Chiv3', 'Chiv4', 'Chiv5', 'Chiv6', 'Chiv7', 'Chiv8', 'Chiv9', 'Chiv10', 'dchi0', 'dchi1', 'dchi2', 'dchi3', 'dchi4', 'Chiv3c', 'Chiv4c', 'Chiv4pc', 'Chiv3ch', 'Chiv4ch', 'Chiv5ch', 'Chiv6ch', 'knotpv', 'knotp'],
            'Kappa': ['kappa1', 'kappa2', 'kappa3', 'kappam1', 'kappam2', 'kappam3', 'phi'],
            'EState': ['value', 'max', 'min', 'Shev', 'Scar', 'Shal', 'Shet', 'Save', 'Smax', 'Smin', 'DS'],
            'Autocorrelation-moran': ['MATSm1', 'MATSm2', 'MATSm3', 'MATSm4', 'MATSm5', 'MATSm6', 'MATSm7', 'MATSm8',
               'MATSv1', 'MATSv2', 'MATSv3', 'MATSv4', 'MATSv5', 'MATSv6', 'MATSv7', 'MATSv8',
               'MATSe1', 'MATSe2', 'MATSe3', 'MATSe4', 'MATSe5', 'MATSe6', 'MATSe7', 'MATSe8',
               'MATSp1', 'MATSp2', 'MATSp3', 'MATSp4', 'MATSp5', 'MATSp6', 'MATSp7', 'MATSp8',],
            'Autocorrelation-geary': ['GATSm1', 'GATSm2', 'GATSm3', 'GATSm4', 'GATSm5', 'GATSm6', 'GATSm7', 'GATSm8',
               'GATSv1', 'GATSv2', 'GATSv3', 'GATSv4', 'GATSv5', 'GATSv6', 'GATSv7', 'GATSv8',
               'GATSe1', 'GATSe2', 'GATSe3', 'GATSe4', 'GATSe5', 'GATSe6', 'GATSe7', 'GATSe8',
               'GATSp1', 'GATSp2', 'GATSp3', 'GATSp4', 'GATSp5', 'GATSp6', 'GATSp7', 'GATSp8'],
            'Autocorrelation-broto': ['ATSm1', 'ATSm2', 'ATSm3', 'ATSm4', 'ATSm5', 'ATSm6', 'ATSm7', 'ATSm8',
               'ATSv1', 'ATSv2', 'ATSv3', 'ATSv4', 'ATSv5', 'ATSv6', 'ATSv7', 'ATSv8',
               'ATSe1', 'ATSe2', 'ATSe3', 'ATSe4', 'ATSe5', 'ATSe6', 'ATSe7', 'ATSe8',
               'ATSp1', 'ATSp2', 'ATSp3', 'ATSp4', 'ATSp5', 'ATSp6', 'ATSp7', 'ATSp8'],
            'Molecular properties': ['LogP', 'MR', 'LabuteASA', 'TPSA', 'Hy', 'UI'],            
            'Charge': ['SPP', 'LDI', 'Rnc', 'Rpc', 'Mac', 'Tac', 'Mnc', 'Tnc', 'Mpc', 'Tpc', 'Qass', 'QOss', 'QNss', 'QCss', 'QHss', 'Qmin', 'QOmin', 'QNmin', 'QCmin', 'QHmin', 'Qmax', 'QOmax', 'QNmax', 'QCmax', 'QHmax'],
            'Moe-Type descriptors': [],
            'Daylight-type fingerprints': ['topological'],
            'MACCS fingerprints': ['MACCS'],
            'Atom pairs fingerprints': ['atompairs'],
            'Morgan fingerprints': ['morgan'],
            'TopologicalTorsion fingerprints': ['torsions'],
            'E-state fingerprints': ['Estate'],
        }

        # Analysis
        self.analysis_data_file = None
        self.analysis_data = None
        self.analysis_type = None
        self.analysis_selected_algorithm = None
        self.analysis_status = False
        self.analysis_default_para = {
            'nclusters': 2,
            'n_components': 2,
            'expand_factor': 2,
            'inflate_factor': 2.0,
            'multiply_factor': 2.0,
            'n_components': 5,            
        }
        self.analysis_message_signal.connect(self.analysis_display_message)
        self.analysis_display_signal.connect(self.set_analysis_table_content)

        # Plot
        self.plot_data_file = None
        self.plot_data = None
        self.selected_plot_type = None


        # initialize UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iFeatureOmega')
        self.resize(800, 600)
        self.setWindowIcon(QIcon(os.path.join(pPath, 'images', 'logo.ico')))
        self.setWindowState(Qt.WindowMaximized)
        self.setFont(QFont('Arial', 12))

        """ QWidget """
        self.protein_widget = QWidget()
        self.dna_widget = QWidget()
        self.rna_widget = QWidget()
        self.structure_widget = QWidget()
        self.chemical_widget = QWidget()
        self.analysis_widget = QWidget()
        self.plot_widget = QWidget()
        self.addTab(self.protein_widget, '  Protein  ')
        self.addTab(self.dna_widget, '  DNA  ')
        self.addTab(self.rna_widget, '  RNA  ')
        self.addTab(self.structure_widget, '  Structure  ')
        self.addTab(self.chemical_widget, '  Ligand  ')
        self.addTab(self.analysis_widget, '  Feature analysis  ')
        self.addTab(self.plot_widget, ' Plot ')

        """ Initialize tab """
        self.setup_protein_widgetUI()
        self.setup_dna_widgetUI()
        self.setup_rna_widgetUI()
        self.setup_structure_widgetUI()
        self.setup_chemical_widgetUI()
        self.setup_analysis_widgetUI()
        self.setup_plot_widgetUI()

    """ setup tab UI """
    def setup_protein_widgetUI(self):
        # file
        topGroupBox = QGroupBox('Choose file in FASTA format', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBoxLayout = QHBoxLayout()
        self.protein_file_lineEdit = QLineEdit()
        self.protein_file_lineEdit.setFont(QFont('Arial', 8))
        self.protein_file_button = QPushButton('Open')
        self.protein_file_button.setFont(QFont('Arial', 10))
        self.protein_file_button.clicked.connect(self.get_fasta_file_name)
        topGroupBoxLayout.addWidget(self.protein_file_lineEdit)
        topGroupBoxLayout.addWidget(self.protein_file_button)
        topGroupBox.setLayout(topGroupBoxLayout)

        # encoding list -> treeGroupBox
        treeGroupBox = QGroupBox('Select descriptors', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.protein_desc_treeWidget = QTreeWidget()
        self.protein_desc_treeWidget.setColumnCount(2)
        self.protein_desc_treeWidget.setMinimumWidth(300)
        self.protein_desc_treeWidget.setColumnWidth(0, 150)
        self.protein_desc_treeWidget.setFont(QFont('Arial', 8))
        self.protein_desc_treeWidget.setHeaderLabels(['Codings', 'Definition'])
        self.protein_desc_treeWidget.clicked.connect(self.protein_desc_tree_clicked)
        """ Protein descriptors """
        self.Protein = QTreeWidgetItem(self.protein_desc_treeWidget)
        self.Protein.setExpanded(True)  # set node expanded
        self.Protein.setDisabled(True)
        self.Protein.setText(0, 'Protein')
        self.AAC = QTreeWidgetItem(self.Protein)
        self.AAC.setText(0, 'AAC')
        self.AAC.setText(1, 'Amino Acids Content')
        self.AAC.setCheckState(0, Qt.Unchecked)
        self.AAC.setToolTip(1, 'The AAC encoding calculates the frequency of each amino acid\n type in a protein or peptide sequence.')
        self.EAAC = QTreeWidgetItem(self.Protein)
        self.EAAC.setText(0, 'EAAC')
        self.EAAC.setText(1, 'Enhanced Amino Acids Content')
        self.EAAC.setCheckState(0, Qt.Unchecked)
        self.EAAC.setToolTip(1, 'The EAAC feature calculates the AAC based on the sequence window\n of fixed length that continuously slides from the N- to\n C-terminus of each peptide and can be usually applied to\n encode the peptides with an equal length.')
        CKSAAP = QTreeWidgetItem(self.Protein)
        CKSAAP.setText(0, 'CKSAAP')
        CKSAAP.setText(1, 'Composition of k-spaced Amino Acid Pairs')
        CKSAAP.setCheckState(0, Qt.Unchecked)
        CKSAAP.setToolTip(1, 'The CKSAAP feature encoding calculates the frequency of amino\n acid pairs separated by any k residues.')
        self.DPC = QTreeWidgetItem(self.Protein)
        self.DPC.setText(0, 'DPC')
        self.DPC.setText(1, 'Di-Peptide Composition')
        self.DPC.setCheckState(0, Qt.Unchecked)
        self.DPC.setToolTip(1, 'The DPC descriptor calculate the frequency of di-peptides.')
        DDE = QTreeWidgetItem(self.Protein)
        DDE.setText(0, 'DDE')
        DDE.setText(1, 'Dipeptide Deviation from Expected Mean')
        DDE.setCheckState(0, Qt.Unchecked)
        DDE.setToolTip(1, 'The Dipeptide Deviation from Expected Mean feature vector is\n constructed by computing three parameters, i.e. dipeptide composition (Dc),\n theoretical mean (Tm), and theoretical variance (Tv).')
        self.TPC = QTreeWidgetItem(self.Protein)
        self.TPC.setText(0, 'TPC')
        self.TPC.setText(1, 'Tripeptide Composition')
        self.TPC.setCheckState(0, Qt.Unchecked)
        self.TPC.setToolTip(1, 'The TPC descriptor calculate the frequency of tri-peptides.')
        self.binary = QTreeWidgetItem(self.Protein)
        self.binary.setText(0, 'binary')
        self.binary.setText(1, 'binary')
        self.binary.setCheckState(0, Qt.Unchecked)
        self.binary.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 20-dimensional binary vector.')
        self.binary_6bit = QTreeWidgetItem(self.Protein)
        self.binary_6bit.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_6bit.setText(0, 'binary_6bit')
        self.binary_6bit.setText(1, 'binary (6 bit)')
        self.binary_6bit.setCheckState(0, Qt.Unchecked)
        self.binary_6bit.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 6-dimensional binary vector.')
        self.binary_5bit_type1 = QTreeWidgetItem(self.Protein)
        self.binary_5bit_type1.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_5bit_type1.setText(0, 'binary_5bit type 1')
        self.binary_5bit_type1.setText(1, 'binary (5 bit type 1)')
        self.binary_5bit_type1.setCheckState(0, Qt.Unchecked)
        self.binary_5bit_type1.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 5-dimensional binary vector.')
        self.binary_5bit_type2 = QTreeWidgetItem(self.Protein)
        self.binary_5bit_type2.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_5bit_type2.setText(0, 'binary_5bit type 2')
        self.binary_5bit_type2.setText(1, 'binary (5 bit type 2)')
        self.binary_5bit_type2.setCheckState(0, Qt.Unchecked)
        self.binary_5bit_type2.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 5-dimensional binary vector.')
        self.binary_3bit_type1 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type1.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type1.setText(0, 'binary_3bit type 1')
        self.binary_3bit_type1.setText(1, 'binary (3 bit type 1 - Hydrophobicity)')
        self.binary_3bit_type1.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type1.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type2 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type2.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type2.setText(0, 'binary_3bit type 2')
        self.binary_3bit_type2.setText(1, 'binary (3 bit type 2 - Normalized Van der Waals volume)')
        self.binary_3bit_type2.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type2.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type3 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type3.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type3.setText(0, 'binary_3bit type 3')
        self.binary_3bit_type3.setText(1, 'binary (3 bit type 3 - Polarity)')
        self.binary_3bit_type3.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type3.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type4 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type4.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type4.setText(0, 'binary_3bit type 4')
        self.binary_3bit_type4.setText(1, 'binary (3 bit type 4 - Polarizibility)')
        self.binary_3bit_type4.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type4.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type5 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type5.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type5.setText(0, 'binary_3bit type 5')
        self.binary_3bit_type5.setText(1, 'binary (3 bit type 5 - Charge)')
        self.binary_3bit_type5.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type5.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type6 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type6.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type6.setText(0, 'binary_3bit type 6')
        self.binary_3bit_type6.setText(1, 'binary (3 bit type 6 - Secondary structures)')
        self.binary_3bit_type6.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type6.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type7 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type7.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type7.setText(0, 'binary_3bit type 7')
        self.binary_3bit_type7.setText(1, 'binary (3 bit type 7 - Solvent accessibility)')
        self.binary_3bit_type7.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type7.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.AESNN3 = QTreeWidgetItem(self.Protein)
        self.AESNN3.setText(0, 'AESNN3')
        self.AESNN3.setText(1, 'Learn from alignments')
        self.AESNN3.setCheckState(0, Qt.Unchecked)
        self.AESNN3.setToolTip(1, 'For this descriptor, each amino acid type is described using\n a three-dimensional vector. Values are taken from the three\n hidden units from the neural network trained on structure alignments.')
        self.GAAC = QTreeWidgetItem(self.Protein)
        self.GAAC.setText(0, 'GAAC')
        self.GAAC.setText(1, 'Grouped Amino Acid Composition')
        self.GAAC.setCheckState(0, Qt.Unchecked)
        self.GAAC.setToolTip(1, 'In the GAAC encoding, the 20 amino acid types are further categorized\n into five classes according to their physicochemical properties. It calculate the frequency for each class.')
        self.EGAAC = QTreeWidgetItem(self.Protein)
        self.EGAAC.setText(0, 'EGAAC')
        self.EGAAC.setText(1, 'Enhanced Grouped Amino Acid Composition')
        self.EGAAC.setCheckState(0, Qt.Unchecked)
        self.EGAAC.setToolTip(1, 'It calculates GAAC in windows of fixed length continuously sliding\n from the N- to C-terminal of each peptide and is usually applied\n to peptides with an equal length.')
        CKSAAGP = QTreeWidgetItem(self.Protein)
        CKSAAGP.setText(0, 'CKSAAGP')
        CKSAAGP.setText(1, 'Composition of k-Spaced Amino Acid Group Pairs')
        CKSAAGP.setCheckState(0, Qt.Unchecked)
        CKSAAGP.setToolTip(1, ' It calculates the frequency of amino acid group pairs separated by any k residues.')
        self.GDPC = QTreeWidgetItem(self.Protein)
        self.GDPC.setText(0, 'GDPC')
        self.GDPC.setText(1, 'Grouped Di-Peptide Composition')
        self.GDPC.setCheckState(0, Qt.Unchecked)
        self.GDPC.setToolTip(1, 'GDPC calculate the frequency of amino acid group pairs.')
        self.GTPC = QTreeWidgetItem(self.Protein)
        self.GTPC.setText(0, 'GTPC')
        self.GTPC.setText(1, 'Grouped Tri-Peptide Composition')
        self.GTPC.setCheckState(0, Qt.Unchecked)
        self.GTPC.setToolTip(1, 'GTPC calculate the frequency of grouped tri-peptides.')
        self.AAIndex = QTreeWidgetItem(self.Protein)
        self.AAIndex.setText(0, 'AAIndex')
        self.AAIndex.setText(1, 'AAIndex')
        self.AAIndex.setCheckState(0, Qt.Unchecked)
        self.AAIndex.setToolTip(1, 'The amino acids is respresented by the physicochemical property value in AAindex database.')
        self.ZScale = QTreeWidgetItem(self.Protein)
        self.ZScale.setText(0, 'ZScale')
        self.ZScale.setText(1, 'ZScale')
        self.ZScale.setCheckState(0, Qt.Unchecked)
        self.ZScale.setToolTip(1, 'Each amino acid is characterized by five physicochemical descriptor variables, which were developed by Sandberg et al. in 1998.')
        self.BLOSUM62 = QTreeWidgetItem(self.Protein)
        self.BLOSUM62.setText(0, 'BLOSUM62')
        self.BLOSUM62.setText(1, 'BLOSUM62')
        self.BLOSUM62.setCheckState(0, Qt.Unchecked)
        self.BLOSUM62.setToolTip(1, 'In this descriptor, the BLOSUM62 matrix is employed to represent the\n protein primary sequence information as the basic feature set.')
        NMBroto = QTreeWidgetItem(self.Protein)
        NMBroto.setText(0, 'NMBroto')
        NMBroto.setText(1, 'Normalized Moreau-Broto Autocorrelation')
        NMBroto.setCheckState(0, Qt.Unchecked)
        NMBroto.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        Moran = QTreeWidgetItem(self.Protein)
        Moran.setText(0, 'Moran')
        Moran.setText(1, 'Moran correlation')
        Moran.setCheckState(0, Qt.Unchecked)
        Moran.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        Geary = QTreeWidgetItem(self.Protein)
        Geary.setText(0, 'Geary')
        Geary.setText(1, 'Geary correlation')
        Geary.setCheckState(0, Qt.Unchecked)
        Geary.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        CTDC = QTreeWidgetItem(self.Protein)
        CTDC.setText(0, 'CTDC')
        CTDC.setText(1, 'Composition')
        CTDC.setCheckState(0, Qt.Unchecked)
        CTDC.setToolTip(1, 'The Composition, Transition and Distribution (CTD) features represent\n the amino acid distribution patterns of a specific structural\n or physicochemical property in a protein or peptide sequence.')
        CTDT = QTreeWidgetItem(self.Protein)
        CTDT.setText(0, 'CTDT')
        CTDT.setText(1, 'Transition')
        CTDT.setCheckState(0, Qt.Unchecked)
        CTDT.setToolTip(1, 'The Composition, Transition and Distribution (CTD) features represent\n the amino acid distribution patterns of a specific structural\n or physicochemical property in a protein or peptide sequence.')
        CTDD = QTreeWidgetItem(self.Protein)
        CTDD.setText(0, 'CTDD')
        CTDD.setText(1, 'Distribution')
        CTDD.setCheckState(0, Qt.Unchecked)
        CTDD.setToolTip(1, 'The Composition, Transition and Distribution (CTD) features represent\n the amino acid distribution patterns of a specific structural\n or physicochemical property in a protein or peptide sequence.')
        CTriad = QTreeWidgetItem(self.Protein)
        CTriad.setText(0, 'CTriad')
        CTriad.setText(1, 'Conjoint Triad')
        CTriad.setCheckState(0, Qt.Unchecked)
        CTriad.setToolTip(1, 'The CTriad considers the properties of one amino acid and its\n vicinal amino acids by regarding any three continuous amino\n acids as a single unit.')
        self.KSCTriad = QTreeWidgetItem(self.Protein)
        self.KSCTriad.setText(0, 'KSCTriad')
        self.KSCTriad.setText(1, 'k-Spaced Conjoint Triad')
        self.KSCTriad.setCheckState(0, Qt.Unchecked)
        self.KSCTriad.setToolTip(1, 'The KSCTriad descriptor is based on the Conjoint CTriad descriptor,\n which not only calculates the numbers of three continuous amino acid units,\n but also considers the continuous amino acid units that are separated by any k residues.')
        SOCNumber = QTreeWidgetItem(self.Protein)
        SOCNumber.setText(0, 'SOCNumber')
        SOCNumber.setText(1, 'Sequence-Order-Coupling Number')
        SOCNumber.setCheckState(0, Qt.Unchecked)
        SOCNumber.setToolTip(1, 'The SOCNumber descriptor consider the sequence order coupling number information.')
        QSOrder = QTreeWidgetItem(self.Protein)
        QSOrder.setText(0, 'QSOrder')
        QSOrder.setText(1, 'Quasi-sequence-order')
        QSOrder.setCheckState(0, Qt.Unchecked)
        QSOrder.setToolTip(1, 'Qsorder descriptor coonsider the quasi sequence order information.')
        PAAC = QTreeWidgetItem(self.Protein)
        PAAC.setText(0, 'PAAC')
        PAAC.setText(1, 'Pseudo-Amino Acid Composition')
        PAAC.setCheckState(0, Qt.Unchecked)
        PAAC.setToolTip(1, 'The PAAC descriptor is a combination of a set of discrete sequence correlation\n factors and the 20 components of the conventional amino acid composition.')
        APAAC = QTreeWidgetItem(self.Protein)
        APAAC.setText(0, 'APAAC')
        APAAC.setText(1, 'Amphiphilic Pseudo-Amino Acid Composition')
        APAAC.setCheckState(0, Qt.Unchecked)
        APAAC.setToolTip(1, 'The descriptor contains 20 + 2 lambda discrete numbers:\n the first 20 numbers are the components of the conventional amino acid composition;\n the next 2 lambda numbers are a set of correlation factors that reflect different\n hydrophobicity and hydrophilicity distribution patterns along a protein chain.')
        self.OPF_10bit = QTreeWidgetItem(self.Protein)
        self.OPF_10bit.setText(0, 'OPF_10bit')
        self.OPF_10bit.setText(1, 'Overlapping Property Features (10 bit)')
        self.OPF_10bit.setCheckState(0, Qt.Unchecked)
        self.OPF_10bit.setToolTip(1, 'For this descriptor, the amino acids are classified into 10 groups based their physicochemical properties.')
        self.OPF_7bit_type1 = QTreeWidgetItem(self.Protein)
        self.OPF_7bit_type1.setText(0, 'OPF_7bit type 1')
        self.OPF_7bit_type1.setText(1, 'Overlapping Property Features (7 bit type 1)')
        self.OPF_7bit_type1.setCheckState(0, Qt.Unchecked)
        self.OPF_7bit_type1.setToolTip(1, 'For this descriptor, the amino acids are classified into 7 groups based their physicochemical properties.')
        self.OPF_7bit_type2 = QTreeWidgetItem(self.Protein)
        self.OPF_7bit_type2.setText(0, 'OPF_7bit type 2')
        self.OPF_7bit_type2.setText(1, 'Overlapping Property Features (7 bit type 2)')
        self.OPF_7bit_type2.setCheckState(0, Qt.Unchecked)
        self.OPF_7bit_type2.setToolTip(1, 'For this descriptor, the amino acids are classified into 7 groups based their physicochemical properties.')
        self.OPF_7bit_type3 = QTreeWidgetItem(self.Protein)
        self.OPF_7bit_type3.setText(0, 'OPF_7bit type 3')
        self.OPF_7bit_type3.setText(1, 'Overlapping Property Features (7 bit type 3)')
        self.OPF_7bit_type3.setCheckState(0, Qt.Unchecked)
        self.OPF_7bit_type3.setToolTip(1, 'For this descriptor, the amino acids are classified into 7 groups based their physicochemical properties.')
        pASDC = QTreeWidgetItem(self.Protein)
        pASDC.setText(0, 'ASDC')
        pASDC.setText(1, 'Adaptive skip dipeptide composition')
        pASDC.setCheckState(0, Qt.Unchecked)
        pASDC.setToolTip(1, 'The adaptive skip dipeptide composition is a modified dipeptide composition,\n which sufficiently considers the correlation information present not only between\n adjacent residues but also between intervening residues.')
        # self.proteinKNN = QTreeWidgetItem(self.Protein)
        # self.proteinKNN.setText(0, 'KNN')
        # self.proteinKNN.setText(1, 'K-nearest neighbor')
        # self.proteinKNN.setCheckState(0, Qt.Unchecked)
        # self.proteinKNN.setToolTip(1, 'The KNN descriptor depicts how much one query sample resembles other samples.')
        DistancePair = QTreeWidgetItem(self.Protein)
        DistancePair.setText(0, 'DistancePair')
        DistancePair.setText(1, 'PseAAC of Distance-Pairs and Reduced Alphabet')
        DistancePair.setCheckState(0, Qt.Unchecked)
        DistancePair.setToolTip(1, 'The descriptor incorporates the amino acid distance pair coupling information \nand the amino acid reduced alphabet profile into the general pseudo amino acid composition vector.')
        self.proteinAC = QTreeWidgetItem(self.Protein)
        self.proteinAC.setText(0, 'AC')
        self.proteinAC.setText(1, 'Auto covariance')
        self.proteinAC.setCheckState(0, Qt.Unchecked)
        self.proteinAC.setToolTip(1, 'The AC descriptor measures the correlation of the same physicochemical \nindex between two amino acids separated by a distance of lag along the sequence. ')
        self.proteinCC = QTreeWidgetItem(self.Protein)
        self.proteinCC.setText(0, 'CC')
        self.proteinCC.setText(1, 'Cross covariance')
        self.proteinCC.setCheckState(0, Qt.Unchecked)
        self.proteinCC.setToolTip(1, 'The CC descriptor measures the correlation of two different physicochemical \nindices between two amino acids separated by lag nucleic acids along the sequence.')
        self.proteinACC = QTreeWidgetItem(self.Protein)
        self.proteinACC.setText(0, 'ACC')
        self.proteinACC.setText(1, 'Auto-cross covariance')
        self.proteinACC.setCheckState(0, Qt.Unchecked)
        self.proteinACC.setToolTip(1, 'The Dinucleotide-based Auto-Cross Covariance (ACC) encoding is a combination of AC and CC encoding.')
        PseKRAAC_type1 = QTreeWidgetItem(self.Protein)
        PseKRAAC_type1.setText(0, 'PseKRAAC type 1')
        PseKRAAC_type1.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 1')
        PseKRAAC_type1.setCheckState(0, Qt.Unchecked)
        PseKRAAC_type1.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type2 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type2.setText(0, 'PseKRAAC type 2')
        self.PseKRAAC_type2.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 2')
        self.PseKRAAC_type2.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type2.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type3A = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type3A.setText(0, 'PseKRAAC type 3A')
        self.PseKRAAC_type3A.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 3A')
        self.PseKRAAC_type3A.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type3A.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type3B = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type3B.setText(0, 'PseKRAAC type 3B')
        self.PseKRAAC_type3B.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 3B')
        self.PseKRAAC_type3B.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type3B.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type4 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type4.setText(0, 'PseKRAAC type 4')
        self.PseKRAAC_type4.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 4')
        self.PseKRAAC_type4.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type4.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type5 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type5.setText(0, 'PseKRAAC type 5')
        self.PseKRAAC_type5.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 5')
        self.PseKRAAC_type5.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type5.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type6A = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type6A.setText(0, 'PseKRAAC type 6A')
        self.PseKRAAC_type6A.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 6A')
        self.PseKRAAC_type6A.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type6A.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type6B = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type6B.setText(0, 'PseKRAAC type 6B')
        self.PseKRAAC_type6B.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 6B')
        self.PseKRAAC_type6B.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type6B.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type6C = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type6C.setText(0, 'PseKRAAC type 6C')
        self.PseKRAAC_type6C.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 6C')
        self.PseKRAAC_type6C.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type6C.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type7 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type7.setText(0, 'PseKRAAC type 7')
        self.PseKRAAC_type7.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 7')
        self.PseKRAAC_type7.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type7.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type8 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type8.setText(0, 'PseKRAAC type 8')
        self.PseKRAAC_type8.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 8')
        self.PseKRAAC_type8.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type8.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type9 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type9.setText(0, 'PseKRAAC type 9')
        self.PseKRAAC_type9.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 9')
        self.PseKRAAC_type9.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type9.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type10 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type10.setText(0, 'PseKRAAC type 10')
        self.PseKRAAC_type10.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 10')
        self.PseKRAAC_type10.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type10.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type11 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type11.setText(0, 'PseKRAAC type 11')
        self.PseKRAAC_type11.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 11')
        self.PseKRAAC_type11.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type11.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type12 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type12.setText(0, 'PseKRAAC type 12')
        self.PseKRAAC_type12.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 12')
        self.PseKRAAC_type12.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type12.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type13 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type13.setText(0, 'PseKRAAC type 13')
        self.PseKRAAC_type13.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 13')
        self.PseKRAAC_type13.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type13.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type14 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type14.setText(0, 'PseKRAAC type 14')
        self.PseKRAAC_type14.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 14')
        self.PseKRAAC_type14.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type14.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type15 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type15.setText(0, 'PseKRAAC type 15')
        self.PseKRAAC_type15.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 15')
        self.PseKRAAC_type15.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type15.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type16 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type16.setText(0, 'PseKRAAC type 16')
        self.PseKRAAC_type16.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 16')
        self.PseKRAAC_type16.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type16.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        treeLayout.addWidget(self.protein_desc_treeWidget)
        treeGroupBox.setLayout(treeLayout)
        
       
        # operation
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.protein_start_button = QPushButton('Start')
        self.protein_start_button.setFont(QFont('Arial', 10))
        self.protein_start_button.clicked.connect(self.run_calculating_protein_descriptors)
        self.protein_desc_slim_button = QPushButton('Show descriptor slims')
        self.protein_desc_slim_button.clicked.connect(self.show_protein_slims)
        self.protein_desc_slim_button.setFont(QFont('Arial', 10))

        startLayout.addWidget(self.protein_start_button)
        startLayout.addWidget(self.protein_desc_slim_button)

        # layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)        
        left_vertical_layout.addWidget(startGroupBox)

        # widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        # QTableWidget
        self.protein_viewWidget = QTabWidget()
        # self.protein_viewWidget.currentChanged.connect(self.protein_displayHtml)
        self.protein_desc_tableWidget = TableWidget.TableWidget()
        # density plot
        self.protein_desc_histWidget = QWidget()
        self.protein_desc_hist_layout = QVBoxLayout(self.protein_desc_histWidget)
        self.protein_desc_histogram = PlotWidgets.HistogramWidget()
        self.protein_desc_hist_layout.addWidget(self.protein_desc_histogram)

        # heatmap
        self.protein_heatmap_widget = QWidget()
        self.protein_heatmap_layout = QVBoxLayout(self.protein_heatmap_widget)
        self.protein_heatmap = PlotWidgets.HeatMapSpanSelector()
        self.protein_heatmap_layout.addWidget(self.protein_heatmap)

        # boxplot
        self.protein_boxplot_widget = QWidget()
        self.protein_boxplot_layout = QVBoxLayout(self.protein_boxplot_widget)
        self.protein_boxplot = PlotWidgets.BoxplotSpanSelector()
        self.protein_boxplot_layout.addWidget(self.protein_boxplot)

        # relations plot
        self.protein_relation_widget = QWidget()
        self.protein_relation_layout = QVBoxLayout(self.protein_relation_widget)
        self.protein_relation = PlotWidgets.CircosWidget()
        self.protein_relation_layout.addWidget(self.protein_relation)

        self.protein_viewWidget.addTab(self.protein_desc_tableWidget, ' Data ')
        self.protein_viewWidget.addTab(self.protein_desc_histWidget, ' Data distribution ')
        self.protein_viewWidget.addTab(self.protein_heatmap_widget, ' Heatmap ')
        self.protein_viewWidget.addTab(self.protein_boxplot_widget, ' Boxplot ')
        self.protein_viewWidget.addTab(self.protein_relation_widget, ' Circular plot ')

        # splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(self.protein_viewWidget)
        splitter_1.setSizes([100, 1200])

        # vertical layout
        vLayout = QVBoxLayout()

        # status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.protein_status_label = QLabel('Welcome to iFeatureOmega.')
        self.protein_progress_bar = QLabel()
        self.protein_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.protein_status_label)
        statusLayout.addWidget(self.protein_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.protein_widget.setLayout(vLayout)

    def setup_dna_widgetUI(self):
        # file
        topGroupBox = QGroupBox('Choose file in FASTA format', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBoxLayout = QHBoxLayout()
        self.dna_file_lineEdit = QLineEdit()
        self.dna_file_lineEdit.setFont(QFont('Arial', 8))
        self.dna_file_button = QPushButton('Open')
        self.dna_file_button.clicked.connect(self.get_dna_file_name)
        self.dna_file_button.setFont(QFont('Arial', 10))
        topGroupBoxLayout.addWidget(self.dna_file_lineEdit)
        topGroupBoxLayout.addWidget(self.dna_file_button)
        topGroupBox.setLayout(topGroupBoxLayout)

        # encoding list -> treeGroupBox
        treeGroupBox = QGroupBox('Select descriptors', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.dna_desc_treeWidget = QTreeWidget()
        self.dna_desc_treeWidget.setColumnCount(2)
        self.dna_desc_treeWidget.setMinimumWidth(300)
        self.dna_desc_treeWidget.setColumnWidth(0, 150)
        self.dna_desc_treeWidget.setFont(QFont('Arial', 8))
        self.dna_desc_treeWidget.setHeaderLabels(['Codings', 'Definition'])
        self.dna_desc_treeWidget.clicked.connect(self.dna_desc_tree_clicked)
        """ DNA descriptors """
        # DNA
        self.DNA = QTreeWidgetItem(self.dna_desc_treeWidget)
        self.DNA.setExpanded(True)
        self.DNA.setDisabled(True)
        self.DNA.setText(0, 'DNA')
        Kmer = QTreeWidgetItem(self.DNA)
        Kmer.setText(0, 'Kmer')
        Kmer.setText(1, 'The occurrence frequencies of k neighboring nucleic acids')
        Kmer.setCheckState(0, Qt.Unchecked)
        Kmer.setToolTip(1, 'For kmer descriptor, the DNA or RNA sequences are represented\n as the occurrence frequencies of k neighboring nucleic acids.')
        RCKmer = QTreeWidgetItem(self.DNA)
        RCKmer.setText(0, 'RCKmer')
        RCKmer.setText(1, 'Reverse Compliment Kmer')
        RCKmer.setCheckState(0, Qt.Unchecked)
        RCKmer.setToolTip(1, 'The RCKmer descriptor is a variant of kmer descriptor,\n in which the kmers are not expected to be strand-specific. ')
        dnaMismatch = QTreeWidgetItem(self.DNA)
        dnaMismatch.setText(0, 'Mismatch')
        dnaMismatch.setText(1, 'Mismatch profile')
        dnaMismatch.setCheckState(0, Qt.Unchecked)
        dnaMismatch.setToolTip(1, 'The mismatch profile also calculates the occurrences of kmers,\n but allows max m inexact matching (m < k).')
        dnaSubsequence = QTreeWidgetItem(self.DNA)
        dnaSubsequence.setText(0, 'Subsequence')
        dnaSubsequence.setText(1, 'Subsequence profile')
        dnaSubsequence.setCheckState(0, Qt.Unchecked)
        dnaSubsequence.setToolTip(1, 'The subsequence descriptor allows non-contiguous matching.')
        self.NAC = QTreeWidgetItem(self.DNA)
        self.NAC.setText(0, 'NAC')
        self.NAC.setText(1, 'Nucleic Acid Composition')
        self.NAC.setCheckState(0, Qt.Unchecked)
        self.NAC.setToolTip(1, 'The NAC encoding calculates the frequency of each nucleic acid type in a nucleotide sequence.')        
        self.ANF = QTreeWidgetItem(self.DNA)
        self.ANF.setText(0, 'ANF')
        self.ANF.setText(1, 'Accumulated Nucleotide Frequency')
        self.ANF.setCheckState(0, Qt.Unchecked)
        self.ANF.setToolTip(1, 'The ANF encoding include the nucleotide frequency information and the distribution of each nucleotide in the RNA sequence.')
        self.ENAC = QTreeWidgetItem(self.DNA)
        self.ENAC.setText(0, 'ENAC')
        self.ENAC.setText(1, 'Enhanced Nucleic Acid Composition')
        self.ENAC.setCheckState(0, Qt.Unchecked)
        self.ENAC.setToolTip(1, 'The ENAC descriptor calculates the NAC based on the sequence window\n of fixed length that continuously slides from the 5\' to 3\' terminus\n of each nucleotide sequence and can be usually applied to encode the\n nucleotide sequence with an equal length.')
        self.DNAbinary = QTreeWidgetItem(self.DNA)
        self.DNAbinary.setText(0, 'binary')
        self.DNAbinary.setText(1, 'DNA binary')
        self.DNAbinary.setCheckState(0, Qt.Unchecked)
        self.DNAbinary.setToolTip(1, 'In the Binary encoding, each amino acid is represented by a 4-dimensional binary vector.')
        self.dnaPS2 = QTreeWidgetItem(self.DNA)
        self.dnaPS2.setText(0, 'PS2')
        self.dnaPS2.setText(1, 'Position-specific of two nucleotides')
        self.dnaPS2.setCheckState(0, Qt.Unchecked)
        self.dnaPS2.setToolTip(1, 'There are 4 x 4 = 16 pairs of adjacent pairwise nucleotides, \nthus a single variable representing one such pair gets one-hot\n (i.e. binary) encoded into 16 binary variables.')
        self.dnaPS3 = QTreeWidgetItem(self.DNA)
        self.dnaPS3.setText(0, 'PS3')
        self.dnaPS3.setText(1, 'Position-specific of three nucleotides')
        self.dnaPS3.setCheckState(0, Qt.Unchecked)
        self.dnaPS3.setToolTip(1, 'The PS3 descriptor is encoded for three adjacent nucleotides in a similar way with PS2.')
        self.dnaPS4 = QTreeWidgetItem(self.DNA)
        self.dnaPS4.setText(0, 'PS4')
        self.dnaPS4.setText(1, 'Position-specific of four nucleotides')
        self.dnaPS4.setCheckState(0, Qt.Unchecked)
        self.dnaPS4.setToolTip(1, 'The PS4 descriptor is encoded for four adjacent nucleotides in a similar way with PS2.')
        CKSNAP = QTreeWidgetItem(self.DNA)
        CKSNAP.setText(0, 'CKSNAP')
        CKSNAP.setText(1, 'Composition of k-spaced Nucleic Acid Pairs')
        CKSNAP.setCheckState(0, Qt.Unchecked)
        CKSNAP.setToolTip(1, 'The CKSNAP feature encoding calculates the frequency of nucleic acid pairs separated by any k nucleic acid.')
        self.NCP = QTreeWidgetItem(self.DNA)
        self.NCP.setText(0, 'NCP')
        self.NCP.setText(1, 'Nucleotide Chemical Property')
        self.NCP.setCheckState(0, Qt.Unchecked)
        self.NCP.setToolTip(1, 'Based on chemical properties, A can be represented by coordinates (1, 1, 1), \nC can be represented by coordinates (0, 1, 0), G can be represented by coordinates (1, 0, 0), \nU can be represented by coordinates (0, 0, 1). ')
        # self.PSTNPss = QTreeWidgetItem(self.DNA)
        # self.PSTNPss.setText(0, 'PSTNPss')
        # self.PSTNPss.setText(1, 'Position-specific trinucleotide propensity based on single-strand')
        # self.PSTNPss.setCheckState(0, Qt.Unchecked)
        # self.PSTNPss.setToolTip(1, 'The PSTNPss descriptor usie a statistical strategy based on single-stranded characteristics of DNA or RNA.')
        # self.PSTNPds = QTreeWidgetItem(self.DNA)
        # self.PSTNPds.setText(0, 'PSTNPds')
        # self.PSTNPds.setText(1, 'Position-specific trinucleotide propensity based on double-strand')
        # self.PSTNPds.setCheckState(0, Qt.Unchecked)
        # self.PSTNPds.setToolTip(1, 'The PSTNPds descriptor use a statistical strategy based on double-stranded characteristics of DNA according to complementary base pairing.')
        self.EIIP = QTreeWidgetItem(self.DNA)
        self.EIIP.setText(0, 'EIIP')
        self.EIIP.setText(1, 'Electron-ion interaction pseudopotentials of trinucleotide')
        self.EIIP.setCheckState(0, Qt.Unchecked)
        self.EIIP.setToolTip(1, 'The EIIP directly use the EIIP value represent the nucleotide in the DNA sequence.')
        PseEIIP = QTreeWidgetItem(self.DNA)
        PseEIIP.setText(0, 'PseEIIP')
        PseEIIP.setText(1, 'Electron-ion interaction pseudopotentials of trinucleotide')
        PseEIIP.setCheckState(0, Qt.Unchecked)
        PseEIIP.setToolTip(1, 'Electron-ion interaction pseudopotentials of trinucleotide.')
        DNAASDC = QTreeWidgetItem(self.DNA)
        DNAASDC.setText(0, 'ASDC')
        DNAASDC.setText(1, 'Adaptive skip dinucleotide composition')
        DNAASDC.setCheckState(0, Qt.Unchecked)
        DNAASDC.setToolTip(1, 'The adaptive skip dipeptide composition is a modified dinucleotide composition, \nwhich sufficiently considers the correlation information present not only between \nadjacent residues but also between intervening residues.')
        self.dnaDBE = QTreeWidgetItem(self.DNA)
        self.dnaDBE.setText(0, 'DBE')
        self.dnaDBE.setText(1, 'Dinucleotide binary encoding')
        self.dnaDBE.setCheckState(0, Qt.Unchecked)
        self.dnaDBE.setToolTip(1, 'The DBE descriptor encapsulates the positional information of the dinucleotide at each position in the sequence.')
        self.dnaLPDF = QTreeWidgetItem(self.DNA)
        self.dnaLPDF.setText(0, 'LPDF')
        self.dnaLPDF.setText(1, 'Local position-specific dinucleotide frequency')
        self.dnaLPDF.setCheckState(0, Qt.Unchecked)
        self.dnaLPDF.setToolTip(1, 'The LPDF descriptor calculate the local position-specific dinucleotide frequency.')
        dnaDPCP = QTreeWidgetItem(self.DNA)
        dnaDPCP.setText(0, 'DPCP')
        dnaDPCP.setText(1, 'Dinucleotide physicochemical properties')
        dnaDPCP.setCheckState(0, Qt.Unchecked)
        dnaDPCP.setToolTip(1, 'The DPCP descriptor calculate the value of frequency of dinucleotide multiplied by dinucleotide physicochemical properties.')
        self.dnaDPCP2 = QTreeWidgetItem(self.DNA)
        self.dnaDPCP2.setText(0, 'DPCP type2')
        self.dnaDPCP2.setText(1, 'Dinucleotide physicochemical properties type 2')
        self.dnaDPCP2.setCheckState(0, Qt.Unchecked)
        self.dnaDPCP2.setToolTip(1, 'The DPCP2 descriptor calculate the position specific dinucleotide physicochemical properties.')
        dnaTPCP = QTreeWidgetItem(self.DNA)
        dnaTPCP.setText(0, 'TPCP')
        dnaTPCP.setText(1, 'Trinucleotide physicochemical properties')
        dnaTPCP.setCheckState(0, Qt.Unchecked)
        dnaTPCP.setToolTip(1, 'The TPCP descriptor calculate the value of frequency of trinucleotide multiplied by trinucleotide physicochemical properties.')
        self.dnaTPCP2 = QTreeWidgetItem(self.DNA)
        self.dnaTPCP2.setText(0, 'TPCP type2')
        self.dnaTPCP2.setText(1, 'Trinucleotide physicochemical properties type 2')
        self.dnaTPCP2.setCheckState(0, Qt.Unchecked)
        self.dnaTPCP2.setToolTip(1, 'The TPCP2 descriptor calculate the position specific trinucleotide physicochemical properties.')
        dnaMMI = QTreeWidgetItem(self.DNA)
        dnaMMI.setText(0, 'MMI')
        dnaMMI.setText(1, 'Multivariate mutual information')
        dnaMMI.setCheckState(0, Qt.Unchecked)
        # self.dnaKNN = QTreeWidgetItem(self.DNA)
        # self.dnaKNN.setText(0, 'KNN')
        # self.dnaKNN.setText(1, 'K-nearest neighbor')
        # self.dnaKNN.setCheckState(0, Qt.Unchecked)
        # self.dnaKNN.setToolTip(1, 'The K-nearest neighbor descriptor depicts how much one query sample resembles other samples.')
        dnazcurve9bit = QTreeWidgetItem(self.DNA)
        dnazcurve9bit.setText(0, 'Z_curve_9bit')
        dnazcurve9bit.setText(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides')
        dnazcurve9bit.setCheckState(0, Qt.Unchecked)
        dnazcurve9bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides.')
        self.dnazcurve12bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve12bit.setText(0, 'Z_curve_12bit')
        self.dnazcurve12bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.dnazcurve12bit.setCheckState(0, Qt.Unchecked)
        self.dnazcurve12bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.dnazcurve36bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve36bit.setText(0, 'Z_curve_36bit')
        self.dnazcurve36bit.setText(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.dnazcurve36bit.setCheckState(0, Qt.Unchecked)
        self.dnazcurve36bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.dnazcurve48bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve48bit.setText(0, 'Z_curve_48bit')
        self.dnazcurve48bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.dnazcurve48bit.setCheckState(0, Qt.Unchecked)
        self.dnazcurve48bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.dnazcurve144bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve144bit.setText(0, 'Z_curve_144bit')
        self.dnazcurve144bit.setText(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        self.dnazcurve144bit.setCheckState(0, Qt.Unchecked)
        self.dnazcurve144bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        dnaNMBroto = QTreeWidgetItem(self.DNA)
        dnaNMBroto.setText(0, 'NMBroto')
        dnaNMBroto.setText(1, 'Normalized Moreau-Broto Autocorrelation')
        dnaNMBroto.setCheckState(0, Qt.Unchecked)
        dnaNMBroto.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        dnaMoran = QTreeWidgetItem(self.DNA)
        dnaMoran.setText(0, 'Moran')
        dnaMoran.setText(1, 'Moran correlation')
        dnaMoran.setCheckState(0, Qt.Unchecked)
        dnaMoran.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        dnaGeary = QTreeWidgetItem(self.DNA)
        dnaGeary.setText(0, 'Geary')
        dnaGeary.setText(1, 'Geary correlation')
        dnaGeary.setCheckState(0, Qt.Unchecked)
        dnaGeary.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        self.DAC = QTreeWidgetItem(self.DNA)
        self.DAC.setText(0, 'DAC')
        self.DAC.setText(1, 'Dinucleotide-based Auto Covariance')
        self.DAC.setCheckState(0, Qt.Unchecked)
        self.DAC.setToolTip(1, 'The DAC descriptor measures the correlation of the same physicochemical \nindex between two dinucleotides separated by a distance of lag along the sequence.')
        self.DCC = QTreeWidgetItem(self.DNA)
        self.DCC.setText(0, 'DCC')
        self.DCC.setText(1, 'Dinucleotide-based Cross Covariance')
        self.DCC.setCheckState(0, Qt.Unchecked)
        self.DCC.setToolTip(1, 'The DCC descriptor measures the correlation of two different physicochemical \nindices between two dinucleotides separated by lag nucleic acids along the sequence.')
        DACC = QTreeWidgetItem(self.DNA)
        DACC.setText(0, 'DACC')
        DACC.setText(1, 'Dinucleotide-based Auto-Cross Covariance')
        DACC.setCheckState(0, Qt.Unchecked)
        DACC.setToolTip(1, 'The DACC encoding is a combination of DAC and DCC encoding.')
        self.TAC = QTreeWidgetItem(self.DNA)
        self.TAC.setText(0, 'TAC')
        self.TAC.setText(1, 'Trinucleotide-based Auto Covariance')
        self.TAC.setCheckState(0, Qt.Unchecked)
        self.TAC.setToolTip(1, 'The TAC descriptor measures the correlation of the same physicochemical \nindex between two trinucleotides separated by a distance of lag along the sequence.')
        self.TCC = QTreeWidgetItem(self.DNA)
        self.TCC.setText(0, 'TCC')
        self.TCC.setText(1, 'Trinucleotide-based Cross Covariance')
        self.TCC.setCheckState(0, Qt.Unchecked)
        self.TCC.setToolTip(1, 'The DCC descriptor measures the correlation of two different physicochemical \nindices between two trinucleotides separated by lag nucleic acids along the sequence.')
        TACC = QTreeWidgetItem(self.DNA)
        TACC.setText(0, 'TACC')
        TACC.setText(1, 'Trinucleotide-based Auto-Cross Covariance')
        TACC.setCheckState(0, Qt.Unchecked)
        TACC.setToolTip(1, 'The TACC encoding is a combination of TAC and TCC encoding.')
        PseDNC = QTreeWidgetItem(self.DNA)
        PseDNC.setText(0, 'PseDNC')
        PseDNC.setText(1, 'Pseudo Dinucleotide Composition')
        PseDNC.setCheckState(0, Qt.Unchecked)
        PseDNC.setToolTip(1, 'The PseDNC encodings incorporate contiguous local sequence-order information and the global sequence-order information into the feature vector of the nucleotide sequence.')
        PseKNC = QTreeWidgetItem(self.DNA)
        PseKNC.setText(0, 'PseKNC')
        PseKNC.setText(1, 'Pseudo k-tupler Composition')
        PseKNC.setCheckState(0, Qt.Unchecked)
        PseKNC.setToolTip(1, 'The PseKNC descriptor incorporate the k-tuple nucleotide composition.')
        PCPseDNC = QTreeWidgetItem(self.DNA)
        PCPseDNC.setText(0, 'PCPseDNC')
        PCPseDNC.setText(1, 'Parallel Correlation Pseudo Dinucleotide Composition')
        PCPseDNC.setCheckState(0, Qt.Unchecked)
        PCPseDNC.setToolTip(1, 'The PCPseDNC descriptor consider parallel correlation pseudo trinucleotide composition information.')
        PCPseTNC = QTreeWidgetItem(self.DNA)
        PCPseTNC.setText(0, 'PCPseTNC')
        PCPseTNC.setText(1, 'Parallel Correlation Pseudo Trinucleotide Composition')
        PCPseTNC.setCheckState(0, Qt.Unchecked)
        PCPseTNC.setToolTip(1, 'The PCPseTNC descriptor consider parallel correlation pseudo trinucleotide composition information.')
        SCPseDNC = QTreeWidgetItem(self.DNA)
        SCPseDNC.setText(0, 'SCPseDNC')
        SCPseDNC.setText(1, 'Series Correlation Pseudo Dinucleotide Composition')
        SCPseDNC.setCheckState(0, Qt.Unchecked)
        SCPseDNC.setToolTip(1, 'The SCPseDNC descriptor consider series correlation pseudo dinucleotide composition information.')
        SCPseTNC = QTreeWidgetItem(self.DNA)
        SCPseTNC.setText(0, 'SCPseTNC')
        SCPseTNC.setText(1, 'Series Correlation Pseudo Trinucleotide Composition')
        SCPseTNC.setCheckState(0, Qt.Unchecked)
        SCPseTNC.setToolTip(1, 'The SCPseTNC descriptor consider series correlation pseudo trinucleotide composition.')
        treeLayout.addWidget(self.dna_desc_treeWidget)
        treeGroupBox.setLayout(treeLayout)
        
       
        # operation
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.dna_start_button = QPushButton('Start')
        self.dna_start_button.clicked.connect(self.run_calculating_dna_descriptors)
        self.dna_start_button.setFont(QFont('Arial', 10))
        self.dna_desc_slim_button = QPushButton('Show descriptor slims')
        self.dna_desc_slim_button.clicked.connect(self.show_dna_slims)
        self.dna_desc_slim_button.setFont(QFont('Arial', 10))

        startLayout.addWidget(self.dna_start_button)
        startLayout.addWidget(self.dna_desc_slim_button)

        # layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)        
        left_vertical_layout.addWidget(startGroupBox)

        # widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        # QTableWidget
        self.dna_viewWidget = QTabWidget()
        self.dna_desc_tableWidget = TableWidget.TableWidget()

        # density plot
        self.dna_desc_histWidget = QWidget()
        self.dna_desc_hist_layout = QVBoxLayout(self.dna_desc_histWidget)
        self.dna_desc_histogram = PlotWidgets.HistogramWidget()
        self.dna_desc_hist_layout.addWidget(self.dna_desc_histogram)

        # heatmap
        self.dna_heatmap_widget = QWidget()
        self.dna_heatmap_layout = QVBoxLayout(self.dna_heatmap_widget)
        self.dna_heatmap = PlotWidgets.HeatMapSpanSelector()
        self.dna_heatmap_layout.addWidget(self.dna_heatmap)

        # boxplot
        self.dna_boxplot_widget = QWidget()
        self.dna_boxplot_layout = QVBoxLayout(self.dna_boxplot_widget)
        self.dna_boxplot = PlotWidgets.BoxplotSpanSelector()
        self.dna_boxplot_layout.addWidget(self.dna_boxplot)

        # relations plot
        self.dna_relation_widget = QWidget()
        self.dna_relation_layout = QVBoxLayout(self.dna_relation_widget)
        self.dna_relation = PlotWidgets.CircosWidget()
        self.dna_relation_layout.addWidget(self.dna_relation)		
		
        self.dna_viewWidget.addTab(self.dna_desc_tableWidget, ' Data ')
        self.dna_viewWidget.addTab(self.dna_desc_histWidget, ' Data distribution ')
        self.dna_viewWidget.addTab(self.dna_heatmap_widget, ' Heatmap ')
        self.dna_viewWidget.addTab(self.dna_boxplot_widget, ' Boxplot ')
        self.dna_viewWidget.addTab(self.dna_relation_widget, ' Relations plot ')


        # splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(self.dna_viewWidget)
        splitter_1.setSizes([100, 1200])

        # vertical layout
        vLayout = QVBoxLayout()

        # status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.dna_status_label = QLabel('Welcome to iLearnPlus Analysis')
        self.dna_progress_bar = QLabel()
        self.dna_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.dna_status_label)
        statusLayout.addWidget(self.dna_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.dna_widget.setLayout(vLayout)

    def setup_rna_widgetUI(self):
        # file
        topGroupBox = QGroupBox('Choose file in FASTA format', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBoxLayout = QHBoxLayout()
        self.rna_file_lineEdit = QLineEdit()
        self.rna_file_lineEdit.setFont(QFont('Arial', 8))
        self.rna_file_button = QPushButton('Open')
        self.rna_file_button.setFont(QFont('Arial', 10))
        self.rna_file_button.clicked.connect(self.get_rna_file_name)
        topGroupBoxLayout.addWidget(self.rna_file_lineEdit)
        topGroupBoxLayout.addWidget(self.rna_file_button)
        topGroupBox.setLayout(topGroupBoxLayout)

        # encoding list -> treeGroupBox
        treeGroupBox = QGroupBox('Select descriptors', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.rna_desc_treeWidget = QTreeWidget()
        self.rna_desc_treeWidget.setColumnCount(2)
        self.rna_desc_treeWidget.setMinimumWidth(300)
        self.rna_desc_treeWidget.setColumnWidth(0, 150)
        self.rna_desc_treeWidget.setFont(QFont('Arial', 8))        
        self.rna_desc_treeWidget.setHeaderLabels(['Codings', 'Definition'])
        self.rna_desc_treeWidget.clicked.connect(self.rna_desc_tree_clicked)
        """ RNA descriptors """
        self.RNA = QTreeWidgetItem(self.rna_desc_treeWidget)
        self.RNA.setExpanded(True)
        self.RNA.setText(0, 'RNA')
        self.RNA.setDisabled(True)
        RNAKmer = QTreeWidgetItem(self.RNA)
        RNAKmer.setText(0, 'Kmer')
        RNAKmer.setText(1, 'The occurrence frequencies of k neighboring nucleic acids')
        RNAKmer.setCheckState(0, Qt.Unchecked)
        RNAKmer.setToolTip(1, 'For kmer descriptor, the DNA or RNA sequences are represented\n as the occurrence frequencies of k neighboring nucleic acids.')
        rnaMismatch = QTreeWidgetItem(self.RNA)
        rnaMismatch.setText(0, 'Mismatch')
        rnaMismatch.setText(1, 'Mismatch profile')
        rnaMismatch.setCheckState(0, Qt.Unchecked)
        rnaMismatch.setToolTip(1, 'The mismatch profile also calculates the occurrences of kmers,\n but allows max m inexact matching (m < k).')
        rnaSubsequence = QTreeWidgetItem(self.RNA)
        rnaSubsequence.setText(0, 'Subsequence')
        rnaSubsequence.setText(1, 'Subsequence profile')
        rnaSubsequence.setCheckState(0, Qt.Unchecked)
        rnaSubsequence.setToolTip(1, 'The subsequence descriptor allows non-contiguous matching.')
        self.RNANAC = QTreeWidgetItem(self.RNA)
        self.RNANAC.setText(0, 'NAC')
        self.RNANAC.setText(1, 'Nucleic Acid Composition')
        self.RNANAC.setCheckState(0, Qt.Unchecked)
        self.RNANAC.setToolTip(1, 'The NAC encoding calculates the frequency of each nucleic acid type in a nucleotide sequence.')
        self.RNAENAC = QTreeWidgetItem(self.RNA)
        self.RNAENAC.setText(0, 'ENAC')
        self.RNAENAC.setText(1, 'Enhanced Nucleic Acid Composition')
        self.RNAENAC.setCheckState(0, Qt.Unchecked)
        self.RNAENAC.setToolTip(1, 'The ENAC descriptor calculates the NAC based on the sequence window\n of fixed length that continuously slides from the 5\' to 3\' terminus\n of each nucleotide sequence and can be usually applied to encode the\n nucleotide sequence with an equal length.')
        self.RNAANF = QTreeWidgetItem(self.RNA)
        self.RNAANF.setText(0, 'ANF')
        self.RNAANF.setText(1, 'Accumulated Nucleotide Frequency')
        self.RNAANF.setCheckState(0, Qt.Unchecked)
        self.RNAANF.setToolTip(1, 'The ANF encoding include the nucleotide frequency information and the distribution of each nucleotide in the RNA sequence.')
        self.RNANCP = QTreeWidgetItem(self.RNA)
        self.RNANCP.setText(0, 'NCP')
        self.RNANCP.setText(1, 'Nucleotide Chemical Property')
        self.RNANCP.setCheckState(0, Qt.Unchecked)
        self.RNANCP.setToolTip(1, 'Based on chemical properties, A can be represented by coordinates (1, 1, 1), \nC can be represented by coordinates (0, 1, 0), G can be represented by coordinates (1, 0, 0), \nU can be represented by coordinates (0, 0, 1). ')
        # self.RNAPSTNPss = QTreeWidgetItem(self.RNA)
        # self.RNAPSTNPss.setText(0, 'PSTNPss')
        # self.RNAPSTNPss.setText(1, 'Position-specific trinucleotide propensity based on single-strand')
        # self.RNAPSTNPss.setCheckState(0, Qt.Unchecked)
        # self.RNAPSTNPss.setToolTip(1, 'The PSTNPss descriptor usie a statistical strategy based on single-stranded characteristics of DNA or RNA.')
        self.RNAbinary = QTreeWidgetItem(self.RNA)
        self.RNAbinary.setText(0, 'binary')
        self.RNAbinary.setText(1, 'RNA binary')
        self.RNAbinary.setCheckState(0, Qt.Unchecked)
        self.RNAbinary.setToolTip(1, 'In the Binary encoding, each amino acid is represented by a 4-dimensional binary vector.')
        self.rnaPS2 = QTreeWidgetItem(self.RNA)
        self.rnaPS2.setText(0, 'PS2')
        self.rnaPS2.setText(1, 'Position-specific of two nucleotides')
        self.rnaPS2.setCheckState(0, Qt.Unchecked)
        self.rnaPS2.setToolTip(1, 'There are 4 x 4 = 16 pairs of adjacent pairwise nucleotides, \nthus a single variable representing one such pair gets one-hot\n (i.e. binary) encoded into 16 binary variables.')
        self.rnaPS3 = QTreeWidgetItem(self.RNA)
        self.rnaPS3.setText(0, 'PS3')
        self.rnaPS3.setText(1, 'Position-specific of three nucleotides')
        self.rnaPS3.setCheckState(0, Qt.Unchecked)
        self.rnaPS3.setToolTip(1, 'The PS3 descriptor is encoded for three adjacent nucleotides in a similar way with PS2.')
        self.rnaPS4 = QTreeWidgetItem(self.RNA)
        self.rnaPS4.setText(0, 'PS4')
        self.rnaPS4.setText(1, 'Position-specific of four nucleotides')
        self.rnaPS4.setCheckState(0, Qt.Unchecked)
        self.rnaPS4.setToolTip(1, 'The PS4 descriptor is encoded for four adjacent nucleotides in a similar way with PS2.')
        RNACKSNAP = QTreeWidgetItem(self.RNA)
        RNACKSNAP.setText(0, 'CKSNAP')
        RNACKSNAP.setText(1, 'Composition of k-spaced Nucleic Acid Pairs')
        RNACKSNAP.setCheckState(0, Qt.Unchecked)
        RNACKSNAP.setToolTip(1, 'The CKSNAP feature encoding calculates the frequency of nucleic acid pairs separated by any k nucleic acid.')
        RNAASDC = QTreeWidgetItem(self.RNA)
        RNAASDC.setText(0, 'ASDC')
        RNAASDC.setText(1, 'Adaptive skip di-nucleotide composition')
        RNAASDC.setCheckState(0, Qt.Unchecked)
        RNAASDC.setToolTip(1, 'The adaptive skip dipeptide composition is a modified dinucleotide composition, \nwhich sufficiently considers the correlation information present not only between \nadjacent residues but also between intervening residues.')
        self.rnaDBE = QTreeWidgetItem(self.RNA)
        self.rnaDBE.setText(0, 'DBE')
        self.rnaDBE.setText(1, 'Dinucleotide binary encoding')
        self.rnaDBE.setCheckState(0, Qt.Unchecked)
        self.rnaDBE.setToolTip(1, 'The DBE descriptor encapsulates the positional information of the dinucleotide at each position in the sequence.')
        self.rnaLPDF = QTreeWidgetItem(self.RNA)
        self.rnaLPDF.setText(0, 'LPDF')
        self.rnaLPDF.setText(1, 'Local position-specific dinucleotide frequency')
        self.rnaLPDF.setCheckState(0, Qt.Unchecked)
        self.rnaLPDF.setToolTip(1, 'The LPDF descriptor calculate the local position-specific dinucleotide frequency.')
        rnaDPCP = QTreeWidgetItem(self.RNA)
        rnaDPCP.setText(0, 'DPCP')
        rnaDPCP.setText(1, 'Dinucleotide physicochemical properties')
        rnaDPCP.setCheckState(0, Qt.Unchecked)
        rnaDPCP.setToolTip(1, 'The DPCP descriptor calculate the value of frequency of dinucleotide multiplied by dinucleotide physicochemical properties.')
        self.rnaDPCP2 = QTreeWidgetItem(self.RNA)
        self.rnaDPCP2.setText(0, 'DPCP type2')
        self.rnaDPCP2.setText(1, 'Dinucleotide physicochemical properties type 2')
        self.rnaDPCP2.setCheckState(0, Qt.Unchecked)
        self.rnaDPCP2.setToolTip(1, 'The DPCP2 descriptor calculate the position specific dinucleotide physicochemical properties.')
        rnaMMI = QTreeWidgetItem(self.RNA)
        rnaMMI.setText(0, 'MMI')
        rnaMMI.setText(1, 'Multivariate mutual information')
        rnaMMI.setCheckState(0, Qt.Unchecked)
        rnaMMI.setToolTip(1, 'The MMI descriptor calculate multivariate mutual information on a DNA/RNA sequence.')
        # self.rnaKNN = QTreeWidgetItem(self.RNA)
        # self.rnaKNN.setText(0, 'KNN')
        # self.rnaKNN.setText(1, 'K-nearest neighbor')
        # self.rnaKNN.setCheckState(0, Qt.Unchecked)
        # self.rnaKNN.setToolTip(1, 'The K-nearest neighbor descriptor depicts how much one query sample resembles other samples.')
        rnazcurve9bit = QTreeWidgetItem(self.RNA)
        rnazcurve9bit.setText(0, 'Z_curve_9bit')
        rnazcurve9bit.setText(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides')
        rnazcurve9bit.setCheckState(0, Qt.Unchecked)
        rnazcurve9bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides.')
        self.rnazcurve12bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve12bit.setText(0, 'Z_curve_12bit')
        self.rnazcurve12bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.rnazcurve12bit.setCheckState(0, Qt.Unchecked)
        self.rnazcurve12bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.rnazcurve36bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve36bit.setText(0, 'Z_curve_36bit')
        self.rnazcurve36bit.setText(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.rnazcurve36bit.setCheckState(0, Qt.Unchecked)
        self.rnazcurve36bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.rnazcurve48bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve48bit.setText(0, 'Z_curve_48bit')
        self.rnazcurve48bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.rnazcurve48bit.setCheckState(0, Qt.Unchecked)
        self.rnazcurve48bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.rnazcurve144bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve144bit.setText(0, 'Z_curve_144bit')
        self.rnazcurve144bit.setText(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        self.rnazcurve144bit.setCheckState(0, Qt.Unchecked)
        self.rnazcurve144bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        rnaNMBroto = QTreeWidgetItem(self.RNA)
        rnaNMBroto.setText(0, 'NMBroto')
        rnaNMBroto.setText(1, 'Normalized Moreau-Broto Autocorrelation')
        rnaNMBroto.setCheckState(0, Qt.Unchecked)
        rnaNMBroto.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        rnaMoran = QTreeWidgetItem(self.RNA)
        rnaMoran.setText(0, 'Moran')
        rnaMoran.setText(1, 'Moran correlation')
        rnaMoran.setCheckState(0, Qt.Unchecked)
        rnaMoran.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        rnaGeary = QTreeWidgetItem(self.RNA)
        rnaGeary.setText(0, 'Geary')
        rnaGeary.setText(1, 'Geary correlation')
        rnaGeary.setCheckState(0, Qt.Unchecked)
        rnaGeary.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        self.RNADAC = QTreeWidgetItem(self.RNA)
        self.RNADAC.setText(0, 'DAC')
        self.RNADAC.setText(1, 'Dinucleotide-based Auto Covariance')
        self.RNADAC.setCheckState(0, Qt.Unchecked)
        self.RNADAC.setToolTip(1, 'The DAC descriptor measures the correlation of the same physicochemical \nindex between two dinucleotides separated by a distance of lag along the sequence.')
        self.RNADCC = QTreeWidgetItem(self.RNA)
        self.RNADCC.setText(0, 'DCC')
        self.RNADCC.setText(1, 'Dinucleotide-based Cross Covariance')
        self.RNADCC.setCheckState(0, Qt.Unchecked)
        self.RNADCC.setToolTip(1, 'The DCC descriptor measures the correlation of two different physicochemical \nindices between two dinucleotides separated by lag nucleic acids along the sequence.')
        RNADACC = QTreeWidgetItem(self.RNA)
        RNADACC.setText(0, 'DACC')
        RNADACC.setText(1, 'Dinucleotide-based Auto-Cross Covariance')        
        RNADACC.setCheckState(0, Qt.Unchecked)
        RNADACC.setToolTip(1, 'The DACC encoding is a combination of DAC and DCC encoding.')
        RNAPseDNC = QTreeWidgetItem(self.RNA)
        RNAPseDNC.setText(0, 'PseDNC')
        RNAPseDNC.setText(1, 'Pseudo Nucleic Acid Composition')
        RNAPseDNC.setCheckState(0, Qt.Unchecked)
        RNAPseDNC.setToolTip(1, 'The PseDNC encodings incorporate contiguous local sequence-order information and the global sequence-order information into the feature vector of the nucleotide sequence.')
        RNAPseKNC = QTreeWidgetItem(self.RNA)
        RNAPseKNC.setText(0, 'PseKNC')
        RNAPseKNC.setText(1, 'Pseudo k-tupler Composition')
        RNAPseKNC.setCheckState(0, Qt.Unchecked)
        RNAPseKNC.setToolTip(1, 'The PseKNC descriptor incorporate the k-tuple nucleotide composition.')
        RNAPCPseDNC = QTreeWidgetItem(self.RNA)
        RNAPCPseDNC.setText(0, 'PCPseDNC')
        RNAPCPseDNC.setText(1, 'Parallel Correlation Pseudo Dinucleotide Composition')
        RNAPCPseDNC.setCheckState(0, Qt.Unchecked)
        RNAPCPseDNC.setToolTip(1, 'The PCPseDNC descriptor consider parallel correlation pseudo trinucleotide composition information.')
        RNASCPseDNC = QTreeWidgetItem(self.RNA)
        RNASCPseDNC.setText(0, 'SCPseDNC')
        RNASCPseDNC.setText(1, 'Series Correlation Pseudo Dinucleotide Composition')
        RNASCPseDNC.setCheckState(0, Qt.Unchecked)
        RNASCPseDNC.setToolTip(1, 'The SCPseDNC descriptor consider series correlation pseudo dinucleotide composition information.')

        treeLayout.addWidget(self.rna_desc_treeWidget)
        treeGroupBox.setLayout(treeLayout)
        
       
        # operation
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.rna_start_button = QPushButton('Start')
        self.rna_start_button.clicked.connect(self.run_calculating_rna_descriptors)
        self.rna_start_button.setFont(QFont('Arial', 10))
        self.rna_desc_slim_button = QPushButton('Show descriptor slims')
        self.rna_desc_slim_button.clicked.connect(self.show_rna_slims)
        self.rna_desc_slim_button.setFont(QFont('Arial', 10))

        startLayout.addWidget(self.rna_start_button)
        startLayout.addWidget(self.rna_desc_slim_button)

        # layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)        
        left_vertical_layout.addWidget(startGroupBox)

        # widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        # QTableWidget
        self.rna_viewWidget = QTabWidget()
        self.rna_desc_tableWidget = TableWidget.TableWidget()
        # density plot
        self.rna_desc_histWidget = QWidget()
        self.rna_desc_hist_layout = QVBoxLayout(self.rna_desc_histWidget)
        self.rna_desc_histogram = PlotWidgets.HistogramWidget()
        self.rna_desc_hist_layout.addWidget(self.rna_desc_histogram)

        # heatmap
        self.rna_heatmap_widget = QWidget()
        self.rna_heatmap_layout = QVBoxLayout(self.rna_heatmap_widget)
        self.rna_heatmap = PlotWidgets.HeatMapSpanSelector()
        self.rna_heatmap_layout.addWidget(self.rna_heatmap)

        # boxplot
        self.rna_boxplot_widget = QWidget()
        self.rna_boxplot_layout = QVBoxLayout(self.rna_boxplot_widget)
        self.rna_boxplot = PlotWidgets.BoxplotSpanSelector()
        self.rna_boxplot_layout.addWidget(self.rna_boxplot)

        # relations plot
        self.rna_relation_widget = QWidget()
        self.rna_relation_layout = QVBoxLayout(self.rna_relation_widget)
        self.rna_relation = PlotWidgets.CircosWidget()
        self.rna_relation_layout.addWidget(self.rna_relation)

        self.rna_viewWidget.addTab(self.rna_desc_tableWidget, ' Data ')
        self.rna_viewWidget.addTab(self.rna_desc_histWidget, ' Data distribution ')
        self.rna_viewWidget.addTab(self.rna_heatmap_widget, ' Heatmap ')
        self.rna_viewWidget.addTab(self.rna_boxplot_widget, ' Boxplot ')
        self.rna_viewWidget.addTab(self.rna_relation_widget, ' Relations plot ')

        # splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(self.rna_viewWidget)
        splitter_1.setSizes([100, 1200])

        # vertical layout
        vLayout = QVBoxLayout()

        # status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.rna_status_label = QLabel('Welcome to iLearnPlus Analysis')
        self.rna_progress_bar = QLabel()
        self.rna_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.rna_status_label)
        statusLayout.addWidget(self.rna_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.rna_widget.setLayout(vLayout)

    def setup_structure_widgetUI(self):
        # Open a PDB file
        topGroupBox = QGroupBox('Input', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBoxLayout = QGridLayout()
        # open local pdb file
        self.structure_file_lineEdit = QLineEdit()
        self.structure_file_lineEdit.setFont(QFont('Arial', 8))
        self.structure_file_button = QPushButton('Open')
        self.structure_file_button.clicked.connect(self.upload_pdb_file)
        self.structure_file_button.setFont(QFont('Arial', 10))
        # download from pdb database
        self.pdb_accession_lineEdit = QLineEdit()
        self.pdb_accession_lineEdit.setFont(QFont('Arial', 8))
        self.download_button = QPushButton('Download')
        self.download_button.setFont(QFont('Arial', 10))
        self.download_button.clicked.connect(self.download_pdb_file)
        topGroupBoxLayout.addWidget(QLabel('Open PDB file:'), 1, 0)
        topGroupBoxLayout.addWidget(self.structure_file_lineEdit, 1, 1)
        topGroupBoxLayout.addWidget(self.structure_file_button, 1, 2)
        topGroupBoxLayout.addWidget(QLabel('Or enter PDB ID:'), 2, 0)
        topGroupBoxLayout.addWidget(self.pdb_accession_lineEdit, 2, 1)
        topGroupBoxLayout.addWidget(self.download_button, 2, 2)
        topGroupBox.setLayout(topGroupBoxLayout)        

        # encoding list -> treeGroupBox
        treeGroupBox = QGroupBox('Select descriptors', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.structure_desc_treeWidget = QTreeWidget()
        self.structure_desc_treeWidget.setColumnCount(2)
        self.structure_desc_treeWidget.setMinimumWidth(300)
        self.structure_desc_treeWidget.setColumnWidth(0, 150)
        self.structure_desc_treeWidget.setFont(QFont('Arial', 8))
        self.structure_desc_treeWidget.setHeaderLabels(['Codings', 'Definition'])
        self.structure_desc_treeWidget.clicked.connect(self.structure_desc_tree_clicked)
        """ RNA descriptors """
        self.Structure = QTreeWidgetItem(self.structure_desc_treeWidget)
        self.Structure.setExpanded(True)
        self.Structure.setDisabled(True)
        self.Structure.setText(0, 'Protein structure')
        AAC_type1 = QTreeWidgetItem(self.Structure)
        AAC_type1.setText(0, 'AAC_type1')
        AAC_type1.setText(1, 'Amino acids content type 1')        
        AAC_type2 = QTreeWidgetItem(self.Structure)
        AAC_type2.setText(0, 'AAC_type2')
        AAC_type2.setText(1, 'Amino acids content type 2')        
        GAAC_type1 = QTreeWidgetItem(self.Structure)
        GAAC_type1.setText(0, 'GAAC_type1')
        GAAC_type1.setText(1, 'Grouped amino acids content type 1')        
        GAAC_type2 = QTreeWidgetItem(self.Structure)
        GAAC_type2.setText(0, 'GAAC_type2')
        GAAC_type2.setText(1, 'Grouped amino acids content type 2')        
        SS3_type1 = QTreeWidgetItem(self.Structure)
        SS3_type1.setText(0, 'SS3_type1')
        SS3_type1.setText(1, 'Secondary structure elements type 1')        
        SS3_type2 = QTreeWidgetItem(self.Structure)
        SS3_type2.setText(0, 'SS3_type2')
        SS3_type2.setText(1, 'Secondary structure elements type 2')        
        SS8_type1 = QTreeWidgetItem(self.Structure)
        SS8_type1.setText(0, 'SS8_type1')
        SS8_type1.setText(1, 'Secondary structure elements type 1')        
        SS8_type2 = QTreeWidgetItem(self.Structure)
        SS8_type2.setText(0, 'SS8_type2')
        SS8_type2.setText(1, 'Secondary structure elements type 2')        
        HSE_CA = QTreeWidgetItem(self.Structure)
        HSE_CA.setText(0, 'HSE_CA')
        HSE_CA.setText(1, '')        
        HSE_CB = QTreeWidgetItem(self.Structure)
        HSE_CB.setText(0, 'HSE_CB')
        HSE_CB.setText(1, '')        
        Residue_depth = QTreeWidgetItem(self.Structure)
        Residue_depth.setText(0, 'Residue depth')
        Residue_depth.setText(1, 'Residue depth')        
        AC_type1 = QTreeWidgetItem(self.Structure)
        AC_type1.setText(0, 'AC_type1')
        AC_type1.setText(1, 'Atom content type 1')        
        AC_type2 = QTreeWidgetItem(self.Structure)
        AC_type2.setText(0, 'AC_type2')
        AC_type2.setText(1, 'Atom content type 2')        
        network_index = QTreeWidgetItem(self.Structure)
        network_index.setText(0, 'Network-based index')
        network_index.setText(1, 'Network-based index')

        treeLayout.addWidget(self.structure_desc_treeWidget)
        treeGroupBox.setLayout(treeLayout)        
       
        # operation
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.structure_start_button = QPushButton('Start')
        self.structure_start_button.clicked.connect(self.run_calculating_structure_descriptors)
        self.structure_start_button.setFont(QFont('Arial', 10))        
        startLayout.addWidget(self.structure_start_button)        

        # layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)        
        left_vertical_layout.addWidget(startGroupBox)

        # widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        # QTableWidget
        self.structure_viewWidget = QTabWidget()
        self.structure_desc_tableWidget = TableWidget.TableWidget()
        # density plot
        self.structure_desc_histWidget = QWidget()
        self.structure_desc_hist_layout = QVBoxLayout(self.structure_desc_histWidget)
        self.structure_desc_histogram = PlotWidgets.HistogramWidget()
        self.structure_desc_hist_layout.addWidget(self.structure_desc_histogram)

        # heatmap
        self.structure_heatmap_widget = QWidget()
        self.structure_heatmap_layout = QVBoxLayout(self.structure_heatmap_widget)
        self.structure_heatmap = PlotWidgets.HeatMapSpanSelector()
        self.structure_heatmap_layout.addWidget(self.structure_heatmap)

        # boxplot
        self.structure_boxplot_widget = QWidget()
        self.structure_boxplot_layout = QVBoxLayout(self.structure_boxplot_widget)
        self.structure_boxplot = PlotWidgets.BoxplotSpanSelector()
        self.structure_boxplot_layout.addWidget(self.structure_boxplot)

        # relations plot
        self.structure_relation_widget = QWidget()
        self.structure_relation_layout = QVBoxLayout(self.structure_relation_widget)
        self.structure_relation = PlotWidgets.CircosWidget()
        self.structure_relation_layout.addWidget(self.structure_relation)

        self.structure_viewWidget.addTab(self.structure_desc_tableWidget, ' Data ')
        self.structure_viewWidget.addTab(self.structure_desc_histWidget, ' Data distribution ')
        self.structure_viewWidget.addTab(self.structure_heatmap_widget, ' Heatmap ')
        self.structure_viewWidget.addTab(self.structure_boxplot_widget, ' Boxplot ')
        self.structure_viewWidget.addTab(self.structure_relation_widget, ' Relations plot ')

        # splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(self.structure_viewWidget)
        splitter_1.setSizes([100, 1200])

        # vertical layout
        vLayout = QVBoxLayout()

        # status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.structure_status_label = QLabel('Welcome to iFeatureOmega.')
        self.structure_progress_bar = QLabel()
        self.structure_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.structure_status_label)
        statusLayout.addWidget(self.structure_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.structure_widget.setLayout(vLayout)

    def setup_chemical_widgetUI(self):
        # Input SMILES
        topGroupBox = QGroupBox('Choose chemical format', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBoxLayout = QGridLayout()
        # Input SMILES
        self.chemical_file_lineEdit = QLineEdit()
        self.chemical_file_lineEdit.setFont(QFont('Arial', 8))
        self.chemical_file_button = QPushButton('Open')
        self.chemical_file_button.setFont(QFont('Arial', 10))
        self.chemical_file_button.clicked.connect(self.get_smiles_file)
        # Upload file (*.smi, *.sdf)
        self.smi_file_lineEdit = QLineEdit()
        self.smi_file_lineEdit.setFont(QFont('Arial', 8))
        self.smi_file_button = QPushButton('Open')
        self.smi_file_button.setFont(QFont('Arial', 10))
        topGroupBoxLayout.addWidget(QLabel('SMILES format:'), 1, 0)
        topGroupBoxLayout.addWidget(self.chemical_file_lineEdit, 1, 1)
        topGroupBoxLayout.addWidget(self.chemical_file_button, 1, 2)
        # topGroupBoxLayout.addWidget(QLabel('Or SMI/SDF format:'), 2, 0)
        # topGroupBoxLayout.addWidget(self.smi_file_lineEdit, 2, 1)
        # topGroupBoxLayout.addWidget(self.smi_file_button, 2, 2)
        topGroupBox.setLayout(topGroupBoxLayout)
        

        # encoding list -> treeGroupBox
        treeGroupBox = QGroupBox('Select descriptors', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.chemical_desc_treeWidget = QTreeWidget()
        self.chemical_desc_treeWidget.setColumnCount(1)
        self.chemical_desc_treeWidget.setMinimumWidth(300)
        self.chemical_desc_treeWidget.setColumnWidth(0, 150)
        self.chemical_desc_treeWidget.setFont(QFont('Arial', 8))
        self.chemical_desc_treeWidget.setHeaderLabels(['Codings'])
        self.chemical_desc_treeWidget.clicked.connect(self.chemical_desc_tree_clicked)
        """ Chemical descriptors """
        self.Chemical = QTreeWidgetItem(self.chemical_desc_treeWidget)
        self.Chemical.setExpanded(True)
        self.Chemical.setText(0, 'Chemical')
        self.Chemical.setDisabled(True)
        constitution = QTreeWidgetItem(self.Chemical)
        constitution.setText(0, 'Constitution')
        constitution.setCheckState(0, Qt.Unchecked)
        topology = QTreeWidgetItem(self.Chemical)
        topology.setText(0, 'Topology')
        topology.setCheckState(0, Qt.Unchecked)
        connectivity = QTreeWidgetItem(self.Chemical)
        connectivity.setText(0, 'Connectivity')
        connectivity.setCheckState(0, Qt.Unchecked)
        kappa = QTreeWidgetItem(self.Chemical)
        kappa.setText(0, 'Kappa')
        kappa.setCheckState(0, Qt.Unchecked)
        estate = QTreeWidgetItem(self.Chemical)
        estate.setText(0, 'EState')
        estate.setCheckState(0, Qt.Unchecked)
        chemical_moran = QTreeWidgetItem(self.Chemical)
        chemical_moran.setText(0, 'Autocorrelation-moran')
        chemical_moran.setCheckState(0, Qt.Unchecked)
        chemical_geary = QTreeWidgetItem(self.Chemical)
        chemical_geary.setText(0, 'Autocorrelation-geary')
        chemical_geary.setCheckState(0, Qt.Unchecked)
        chemical_broto = QTreeWidgetItem(self.Chemical)
        chemical_broto.setText(0, 'Autocorrelation-broto')
        chemical_broto.setCheckState(0, Qt.Unchecked)
        molecular_properties = QTreeWidgetItem(self.Chemical)
        molecular_properties.setText(0, 'Molecular properties')
        molecular_properties.setCheckState(0, Qt.Unchecked)
        charge = QTreeWidgetItem(self.Chemical)
        charge.setText(0, 'Charge')
        charge.setCheckState(0, Qt.Unchecked)
        moetype = QTreeWidgetItem(self.Chemical)
        moetype.setText(0, 'Moe-Type descriptors')
        moetype.setCheckState(0, Qt.Unchecked)
        daylight_fp = QTreeWidgetItem(self.Chemical)
        daylight_fp.setText(0, 'Daylight-type fingerprints')
        daylight_fp.setCheckState(0, Qt.Unchecked)
        maccs_fp = QTreeWidgetItem(self.Chemical)
        maccs_fp.setText(0, 'MACCS fingerprints')
        maccs_fp.setCheckState(0, Qt.Unchecked)
        atompairs_fp = QTreeWidgetItem(self.Chemical)
        atompairs_fp.setText(0, 'Atom pairs fingerprints')
        atompairs_fp.setCheckState(0, Qt.Unchecked)
        morgan_fp = QTreeWidgetItem(self.Chemical)
        morgan_fp.setText(0, 'Morgan fingerprints')
        morgan_fp.setCheckState(0, Qt.Unchecked)
        topologicaltorsion_fp = QTreeWidgetItem(self.Chemical)
        topologicaltorsion_fp.setText(0, 'TopologicalTorsion fingerprints')
        topologicaltorsion_fp.setCheckState(0, Qt.Unchecked)
        estate_fp = QTreeWidgetItem(self.Chemical)
        estate_fp.setText(0, 'E-state fingerprints')
        estate_fp.setCheckState(0, Qt.Unchecked)
        treeLayout.addWidget(self.chemical_desc_treeWidget)
        treeGroupBox.setLayout(treeLayout)
       
        # operation
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.chemical_start_button = QPushButton('Start')
        self.chemical_start_button.clicked.connect(self.run_calculating_chemical_descriptors)
        self.chemical_start_button.setFont(QFont('Arial', 10))

        startLayout.addWidget(self.chemical_start_button)

        # layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)        
        left_vertical_layout.addWidget(startGroupBox)

        # widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        # QTableWidget
        self.chemical_viewWidget = QTabWidget()
        self.chemical_desc_tableWidget = TableWidget.TableWidget()
        self.chemical_desc_histWidget = QWidget()
        self.chemical_desc_hist_layout = QVBoxLayout(self.chemical_desc_histWidget)
        self.chemical_desc_histogram = PlotWidgets.HistogramWidget()
        self.chemical_desc_hist_layout.addWidget(self.chemical_desc_histogram)

        # heatmap
        self.chemical_heatmap_widget = QWidget()
        self.chemical_heatmap_layout = QVBoxLayout(self.chemical_heatmap_widget)
        self.chemical_heatmap = PlotWidgets.HeatMapSpanSelector()
        self.chemical_heatmap_layout.addWidget(self.chemical_heatmap)

        # boxplot
        self.chemical_boxplot_widget = QWidget()
        self.chemical_boxplot_layout = QVBoxLayout(self.chemical_boxplot_widget)
        self.chemical_boxplot = PlotWidgets.BoxplotSpanSelector()
        self.chemical_boxplot_layout.addWidget(self.chemical_boxplot)

        # relations plot
        self.chemical_relation_widget = QWidget()
        self.chemical_relation_layout = QVBoxLayout(self.chemical_relation_widget)
        self.chemical_relation = PlotWidgets.CircosWidget()
        self.chemical_relation_layout.addWidget(self.chemical_relation)

        self.chemical_viewWidget.addTab(self.chemical_desc_tableWidget, ' Data ')
        self.chemical_viewWidget.addTab(self.chemical_desc_histWidget, ' Data distribution ')
        self.chemical_viewWidget.addTab(self.chemical_heatmap_widget, ' Heatmap ')
        self.chemical_viewWidget.addTab(self.chemical_boxplot_widget, ' Boxplot ')
        self.chemical_viewWidget.addTab(self.chemical_relation_widget, ' Relations plot ')        

        # splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(self.chemical_viewWidget)
        splitter_1.setSizes([100, 1200])

        # vertical layout
        vLayout = QVBoxLayout()

        # status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.chemical_status_label = QLabel('Welcome to iFeatureOmega')
        self.chemical_progress_bar = QLabel()
        self.chemical_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.chemical_status_label)
        statusLayout.addWidget(self.chemical_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.chemical_widget.setLayout(vLayout)

    def setup_analysis_widgetUI(self):
        # file
        topGroupBox = QGroupBox('Load data', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QGridLayout()

        self.analysis_file_lineEdit = QLineEdit()
        self.analysis_file_lineEdit.setFont(QFont('Arial', 8))
        self.analysis_file_button = QPushButton('Open [without sample label]')
        self.analysis_file_button.setToolTip('Data file without sample label.')
        self.analysis_file_button.setFont(QFont('Arial', 10))
        self.analysis_file_button.clicked.connect(self.data_from_file)
        self.analysis_file_lineEdit2 = QLineEdit()
        self.analysis_file_lineEdit2.setFont(QFont('Arial', 8))
        self.analysis_file_button2 = QPushButton('Open [with sample label]')
        self.analysis_file_button2.setToolTip('Data file with sample label in column 1.')
        self.analysis_file_button2.setFont(QFont('Arial', 10))
        self.analysis_file_button2.clicked.connect(self.data_from_file2)

        self.analysis_data_lineEdit = QLineEdit()
        self.analysis_data_lineEdit.setFont(QFont('Arial', 8))
        self.analysis_data_button = QPushButton('Select')
        self.analysis_data_button.clicked.connect(self.data_from_panels)
        self.analysis_datashape = QLabel('Data shape: ')
        topGroupBoxLayout.addWidget(self.analysis_file_lineEdit, 0, 0)
        topGroupBoxLayout.addWidget(self.analysis_file_button, 0, 1)
        topGroupBoxLayout.addWidget(self.analysis_file_lineEdit2, 1, 0)
        topGroupBoxLayout.addWidget(self.analysis_file_button2, 1, 1)        

        topGroupBoxLayout.addWidget(self.analysis_data_lineEdit, 2, 0)
        topGroupBoxLayout.addWidget(self.analysis_data_button, 2, 1)
        topGroupBoxLayout.addWidget(self.analysis_datashape, 3, 0, 1, 2)
        topGroupBox.setLayout(topGroupBoxLayout)

        # tree
        treeGroupBox = QGroupBox('Analysis algorithms', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.analysis_treeWidget = QTreeWidget()
        self.analysis_treeWidget.setColumnCount(2)
        self.analysis_treeWidget.setMinimumWidth(300)
        self.analysis_treeWidget.setColumnWidth(0, 150)
        self.analysis_treeWidget.setFont(QFont('Arial', 8))
        self.analysis_treeWidget.setHeaderLabels(['Methods', 'Definition'])
        self.analysis_treeWidget.clicked.connect(self.analysis_tree_clicked)
        self.clusterMethods = QTreeWidgetItem(self.analysis_treeWidget)
        self.clusterMethods.setExpanded(True)  # set node expanded
        self.clusterMethods.setText(0, 'Cluster algorithms')
        
        kmeans = QTreeWidgetItem(self.clusterMethods)
        kmeans.setText(0, 'kmeans')
        kmeans.setText(1, 'kmeans clustering')
        minikmeans = QTreeWidgetItem(self.clusterMethods)
        minikmeans.setText(0, 'MiniBatchKMeans')
        minikmeans.setText(1, 'MiniBatchKMeans clustering')
        gmm = QTreeWidgetItem(self.clusterMethods)
        gmm.setText(0, 'GM')
        gmm.setText(1, 'Gaussian mixture clustering')
        agg = QTreeWidgetItem(self.clusterMethods)
        agg.setText(0, 'Agglomerative')
        agg.setText(1, 'Agglomerative clustering')
        spectral = QTreeWidgetItem(self.clusterMethods)
        spectral.setText(0, 'Spectral')
        spectral.setText(1, 'Spectral clustering')
        mcl = QTreeWidgetItem(self.clusterMethods)
        mcl.setText(0, 'MCL')
        mcl.setText(1, 'Markov Cluster algorithm')
        hcluster = QTreeWidgetItem(self.clusterMethods)
        hcluster.setText(0, 'hcluster')
        hcluster.setText(1, 'Hierarchical clustering')
        apc = QTreeWidgetItem(self.clusterMethods)
        apc.setText(0, 'APC')
        apc.setText(1, 'Affinity Propagation Clustering')
        meanshift = QTreeWidgetItem(self.clusterMethods)
        meanshift.setText(0, 'meanshift')
        meanshift.setText(1, 'Mean-shift Clustering')
        dbscan = QTreeWidgetItem(self.clusterMethods)
        dbscan.setText(0, 'DBSCAN')
        dbscan.setText(1, 'DBSCAN Clustering')
        self.dimensionReduction = QTreeWidgetItem(self.analysis_treeWidget)
        self.dimensionReduction.setExpanded(True)  # set node expanded
        self.dimensionReduction.setText(0, 'Dimensionality reduction algorithms')
        pca = QTreeWidgetItem(self.dimensionReduction)
        pca.setText(0, 'PCA')
        pca.setText(1, 'Principal component analysis')
        tsne = QTreeWidgetItem(self.dimensionReduction)
        tsne.setText(0, 't_SNE')
        tsne.setText(1, 't-distributed Stochastic Neighbor Embedding')
        lda = QTreeWidgetItem(self.dimensionReduction)
        lda.setText(0, 'LDA')
        lda.setText(1, 'Latent Dirichlet Allocation')
        self.normalizationMethods = QTreeWidgetItem(self.analysis_treeWidget)
        self.normalizationMethods.setExpanded(True)  # set node expanded
        self.normalizationMethods.setText(0, 'Feature normalization algorithms')
        ZScore = QTreeWidgetItem(self.normalizationMethods)
        ZScore.setText(0, 'ZScore')
        ZScore.setText(1, 'ZScore')
        MinMax = QTreeWidgetItem(self.normalizationMethods)
        MinMax.setText(0, 'MinMax')
        MinMax.setText(1, 'MinMax')
        treeLayout.addWidget(self.analysis_treeWidget)
        treeGroupBox.setLayout(treeLayout)        

        ## parameter
        paraGroupBox = QGroupBox('Parameters', self)
        paraGroupBox.setMaximumHeight(150)
        paraGroupBox.setFont(QFont('Arial', 10))
        paraLayout = QFormLayout(paraGroupBox)
        self.analysis_analysisType_lineEdit = QLineEdit()
        self.analysis_analysisType_lineEdit.setFont(QFont('Arial', 8))
        self.analysis_analysisType_lineEdit.setEnabled(False)
        paraLayout.addRow('Analysis:', self.analysis_analysisType_lineEdit)
        self.analysis_algorithm_lineEdit = QLineEdit()
        self.analysis_algorithm_lineEdit.setFont(QFont('Arial', 8))
        self.analysis_algorithm_lineEdit.setEnabled(False)
        paraLayout.addRow('Algorithm:', self.analysis_algorithm_lineEdit)
        self.analysis_para_lineEdit = QLineEdit()
        self.analysis_para_lineEdit.setFont(QFont('Arial', 8))
        self.analysis_para_lineEdit.setEnabled(False)
        paraLayout.addRow('Parameter(s):', self.analysis_para_lineEdit)

        ## start button
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.analysis_start_button = QPushButton('Start')        
        self.analysis_start_button.setFont(QFont('Arial', 10))
        self.analysis_start_button.clicked.connect(self.run_data_analysis)
        self.analysis_save_button = QPushButton('Save txt')
        self.analysis_save_button.setFont(QFont('Arial', 10))
                     
        startLayout.addWidget(self.analysis_start_button)
        startLayout.addWidget(self.analysis_save_button)
        
        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region        
        self.analysis_tableWidget = QTableWidget()
        self.analysis_tableWidget.setFont(QFont('Arial', 8))
        self.analysis_diagram = PlotWidgets.ClusteringDiagramMatplotlib()
        self.analysis_diagram_widget = QWidget()
        self.analysis_diagram_layout = QVBoxLayout(self.analysis_diagram_widget)
        self.analysis_diagram_layout.addWidget(self.analysis_diagram)

        self.analysis_tableWidget_1 = TableWidget.TableWidget()
        self.analysis_histWidget = QWidget()
        self.analysis_hist_layout = QVBoxLayout(self.analysis_histWidget)
        self.analysis_histogram = PlotWidgets.HistogramWidget()
        self.analysis_hist_layout.addWidget(self.analysis_histogram)
        
        self.analysis_tabWidget = QTabWidget()
        self.analysis_tabWidget.addTab(self.analysis_tableWidget, ' Cluster/Dimensionality reduction result ')
        self.analysis_tabWidget.addTab(self.analysis_diagram_widget, ' Scatter plot ')
        self.analysis_tabWidget.addTab(self.analysis_tableWidget_1, ' Normalizated data ')
        self.analysis_tabWidget.addTab(self.analysis_histWidget, ' Normalizated data distribution ')                
        
        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(self.analysis_tabWidget)
        splitter_1.setSizes([100, 1200])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.analysis_status_label = QLabel('Welcome to iLearnOmega.')
        self.analysis_progress_bar = QLabel()
        self.analysis_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.analysis_status_label)
        statusLayout.addWidget(self.analysis_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.analysis_widget.setLayout(vLayout)

    def setup_plot_widgetUI(self):
        # file
        topGroupBox = QGroupBox('Load data', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QGridLayout()

        self.plot_file_lineEdit = QLineEdit()
        self.plot_file_lineEdit.setFont(QFont('Arial', 8))
        self.plot_file_button = QPushButton('Open [without sample label]')
        self.plot_file_button.setToolTip('Data file without sample label.')
        self.plot_file_button.setFont(QFont('Arial', 10))
        self.plot_file_button.clicked.connect(self.plot_data_from_file)
        self.plot_file_lineEdit2 = QLineEdit()
        self.plot_file_lineEdit2.setFont(QFont('Arial', 8))
        self.plot_file_button2 = QPushButton('Open [with sample label]')
        self.plot_file_button2.setToolTip('Data file with sample label in column 1.')
        self.plot_file_button2.setFont(QFont('Arial', 10))
        self.plot_file_button2.clicked.connect(self.plot_data_from_file2)
       
        self.plot_datashape = QLabel('Data shape: ')
        topGroupBoxLayout.addWidget(self.plot_file_lineEdit, 0, 0)
        topGroupBoxLayout.addWidget(self.plot_file_button, 0, 1)
        topGroupBoxLayout.addWidget(self.plot_file_lineEdit2, 1, 0)
        topGroupBoxLayout.addWidget(self.plot_file_button2, 1, 1)
        topGroupBoxLayout.addWidget(self.plot_datashape, 2, 0, 1, 2)
        topGroupBox.setLayout(topGroupBoxLayout)

        # tree
        treeGroupBox = QGroupBox('Plot types', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.plot_treeWidget = QTreeWidget()
        self.plot_treeWidget.setColumnCount(1)
        self.plot_treeWidget.setMinimumWidth(300)
        self.plot_treeWidget.setColumnWidth(0, 150)
        self.plot_treeWidget.setFont(QFont('Arial', 8))
        self.plot_treeWidget.setHeaderLabels([' '])
        self.plot_treeWidget.clicked.connect(self.plot_tree_clicked)
        
        self.plotTypes = QTreeWidgetItem(self.plot_treeWidget)
        self.plotTypes.setExpanded(True)  # set node expanded
        self.plotTypes.setText(0, 'Plot types')
        hist = QTreeWidgetItem(self.plotTypes)
        hist.setText(0, 'Histogram and kernel density plot')        
        boxplot = QTreeWidgetItem(self.plotTypes)
        boxplot.setText(0, 'Box plot')
        heatmap = QTreeWidgetItem(self.plotTypes)
        heatmap.setText(0, 'Heatmap')
        scatter = QTreeWidgetItem(self.plotTypes)
        scatter.setText(0, 'Scatter plot')
        circular = QTreeWidgetItem(self.plotTypes)
        circular.setText(0, 'Circular plot')
        treeLayout.addWidget(self.plot_treeWidget)
        treeGroupBox.setLayout(treeLayout)

        ## start button
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.plot_start_button = QPushButton('Draw plot')        
        self.plot_start_button.setFont(QFont('Arial', 10))
        self.plot_start_button.clicked.connect(self.draw_plot)
        self.plot_save_button = QPushButton('Save txt')
        self.plot_save_button.setFont(QFont('Arial', 10))
                     
        startLayout.addWidget(self.plot_start_button)
        startLayout.addWidget(self.plot_save_button)
        
        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        # left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region
        self.plot_viewWidget = QWidget()
        self.plot_view_layout = QVBoxLayout(self.plot_viewWidget)
        self.plot_plt = PlotWidgets.HistogramWidget()
        self.plot_view_layout.addWidget(self.plot_plt)
        
        self.plot_tabWidget = QTabWidget()        
        self.plot_tabWidget.addTab(self.plot_viewWidget, ' Plot viewer ')                
        
        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(self.plot_tabWidget)
        splitter_1.setSizes([100, 1200])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.plot_status_label = QLabel('Welcome to iLearnOmega.')
        self.plot_progress_bar = QLabel()
        self.plot_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.plot_status_label)
        statusLayout.addWidget(self.plot_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.plot_widget.setLayout(vLayout)


    """ events in protein widgetUI """
    def protein_display_message(self, message):
        self.protein_status_label.setText(message)

    def recover_protein_treeWidget(self):
        child_num = self.Protein.childCount()
        for i in range(child_num):
            self.Protein.child(i).setCheckState(0, Qt.Unchecked)

    def protein_panel_clear(self):
        self.protein_sequence_file = None
        self.protein_descriptor = None
        self.protein_encodings = None
        self.protein_desc_tableWidget.tableWidget.clear()
        self.protein_file_lineEdit.clear()
        self.Protein.setDisabled(True)
        self.protein_status_label.setText('Welcome to iFeatureOmega.')
        self.recover_protein_treeWidget()
        self.protein_selected_descriptors = set([])

    def check_protein_descriptors(self):
        if self.protein_descriptor is not None and not self.protein_descriptor.is_equal:
            self.EAAC.setDisabled(True)
            self.binary.setDisabled(True)
            self.binary_6bit.setDisabled(True)
            self.binary_5bit_type1.setDisabled(True)
            self.binary_5bit_type2.setDisabled(True)
            self.binary_3bit_type1.setDisabled(True)
            self.binary_3bit_type2.setDisabled(True)
            self.binary_3bit_type3.setDisabled(True)
            self.binary_3bit_type4.setDisabled(True)
            self.binary_3bit_type5.setDisabled(True)
            self.binary_3bit_type6.setDisabled(True)
            self.binary_3bit_type7.setDisabled(True)
            self.AESNN3.setDisabled(True)
            self.EGAAC.setDisabled(True)
            self.AAIndex.setDisabled(True)
            self.ZScale.setDisabled(True)
            self.BLOSUM62.setDisabled(True)
            self.OPF_10bit.setDisabled(True)
            self.OPF_7bit_type1.setDisabled(True)
            self.OPF_7bit_type2.setDisabled(True)
            self.OPF_7bit_type3.setDisabled(True)
            self.proteinKNN.setDisabled(True) 

    def get_fasta_file_name(self):
        self.protein_panel_clear()
        self.protein_sequence_file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'Plain text (*.*)')
        if ok:
            self.protein_file_lineEdit.setText(self.protein_sequence_file)
            self.protein_descriptor = iSequence.Sequence(self.protein_sequence_file)
            if self.protein_descriptor.error_msg != '':
                QMessageBox.critical(self, 'Error', str(self.protein_descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.protein_panel_clear()
            elif self.protein_descriptor.sequence_type != 'Protein':
                QMessageBox.critical(self, 'Error', 'The input seems not protein sequence!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.protein_panel_clear()
            else:
                self.Protein.setDisabled(False)
                self.check_protein_descriptors()
        else:
            self.protein_panel_clear()
    
    def protein_desc_tree_clicked(self, index):
        item = self.protein_desc_treeWidget.currentItem()
        if item and item.text(0) not in ['Protein']:
            if item.checkState(0) == Qt.Unchecked:
                item.setCheckState(0, Qt.Checked)
                self.protein_selected_descriptors.add(item.text(0))
                self.protein_para_setting(item.text(0))
            else:
                item.setCheckState(0, Qt.Unchecked)
                self.protein_selected_descriptors.discard(item.text(0))            

    def protein_para_setting(self, desc):
        if self.protein_descriptor is not None:
            if desc in ['EAAC', 'EGAAC']:
                num, ok = QInputDialog.getInt(self, '%s setting' %desc, 'Sliding window size', 5, 2, 10, 1)
                if ok:
                    self.protein_para_dict[desc]['sliding_window'] = num            
            elif desc in ['CKSAAP', 'CKSAAGP', 'KSCTriad']:
                num, ok = QInputDialog.getInt(self, '%s setting' %desc, 'K-space number', 3, 0, 5, 1)
                if ok:
                    self.protein_para_dict[desc]['kspace'] = num
            elif desc in ['SOCNumber']:
                num, ok = QInputDialog.getInt(self, '%s setting' %desc, 'lag value', 3, 1, self.protein_descriptor.minimum_length_without_minus - 1, 1)
                if ok:
                    self.protein_para_dict[desc]['nlag'] = num
            elif desc in ['QSOrder']:
                lag, weight, ok, = InputDialog.QSOrderInput.getValues(self.protein_descriptor.minimum_length_without_minus - 1)
                if ok:
                    self.protein_para_dict[desc]['nlag'] = lag
                    self.protein_para_dict[desc]['weight'] = weight
            elif desc in ['AAIndex']:
                property, ok = InputDialog.QAAindexInput.getValues()
                if ok:
                    self.protein_para_dict[desc]['aaindex'] = property
            elif desc in ['NMBroto', 'Moran', 'Geary', 'AC', 'CC', 'ACC']:
                lag, property, ok = InputDialog.QAAindex2Input.getValues(self.protein_descriptor.minimum_length_without_minus - 1)
                if ok:
                    self.protein_para_dict[desc]['aaindex'] = property
                    self.protein_para_dict[desc]['nlag'] = lag
            elif desc in ['PAAC', 'APAAC']:
                lambdaValue, weight, ok, = InputDialog.QPAACInput.getValues(self.protein_descriptor.minimum_length_without_minus - 1)
                if ok:
                    self.protein_para_dict[desc]['lambdaValue'] = lambdaValue
                    self.protein_para_dict[desc]['weight'] = weight
            elif desc in ['DistancePair']:
                num, cp, ok = InputDialog.QDistancePairInput.getValues()
                if ok:
                    self.protein_para_dict[desc]['distance'] = num
                    self.protein_para_dict[desc]['cp'] = cp
            elif desc in ['PseKRAAC type 1', 'PseKRAAC type 2', 'PseKRAAC type 3A', 'PseKRAAC type 3B', 'PseKRAAC type 5', 'PseKRAAC type 6A', 'PseKRAAC type 6B', 
                        'PseKRAAC type 6C', 'PseKRAAC type 7', 'PseKRAAC type 8', 'PseKRAAC type 9', 'PseKRAAC type 10', 'PseKRAAC type 11', 'PseKRAAC type 12', 
                        'PseKRAAC type 13', 'PseKRAAC type 14', 'PseKRAAC type 15', 'PseKRAAC type 16']:
                model, gap, lambdaValue, ktuple, clust, ok = InputDialog.QPseKRAACInput.getValues(desc)
                if ok:
                    self.protein_para_dict[desc]['PseKRAAC_model'] = model
                    self.protein_para_dict[desc]['g-gap'] = int(gap)
                    self.protein_para_dict[desc]['lambdaValue'] = int(lambdaValue)
                    self.protein_para_dict[desc]['k-tuple'] = int(ktuple)
                    self.protein_para_dict[desc]['RAAC_clust'] = int(clust)

    def show_protein_slims(self):
        if self.protein_desc_slim_button.text() == 'Show descriptor slims':
            self.AAC.setHidden(True)
            self.DPC.setHidden(True)
            self.TPC.setHidden(True)
            self.binary_6bit.setHidden(True)
            self.binary_5bit_type1.setHidden(True)
            self.binary_5bit_type2.setHidden(True)
            self.binary_3bit_type1.setHidden(True)
            self.binary_3bit_type2.setHidden(True)
            self.binary_3bit_type3.setHidden(True)
            self.binary_3bit_type4.setHidden(True)
            self.binary_3bit_type5.setHidden(True)
            self.binary_3bit_type6.setHidden(True)
            self.binary_3bit_type7.setHidden(True)
            self.GAAC.setHidden(True)
            self.GDPC.setHidden(True)
            self.GTPC.setHidden(True)
            self.KSCTriad.setHidden(True)
            self.OPF_7bit_type1.setHidden(True)
            self.OPF_7bit_type2.setHidden(True)
            self.OPF_7bit_type3.setHidden(True)
            self.proteinAC.setHidden(True)
            self.proteinCC.setHidden(True)
            self.PseKRAAC_type2.setHidden(True)
            self.PseKRAAC_type3A.setHidden(True)
            self.PseKRAAC_type3B.setHidden(True)
            self.PseKRAAC_type4.setHidden(True)
            self.PseKRAAC_type5.setHidden(True)
            self.PseKRAAC_type6A.setHidden(True)
            self.PseKRAAC_type6B.setHidden(True)
            self.PseKRAAC_type6C.setHidden(True)
            self.PseKRAAC_type7.setHidden(True)
            self.PseKRAAC_type8.setHidden(True)
            self.PseKRAAC_type9.setHidden(True)
            self.PseKRAAC_type10.setHidden(True)
            self.PseKRAAC_type11.setHidden(True)
            self.PseKRAAC_type12.setHidden(True)
            self.PseKRAAC_type13.setHidden(True)
            self.PseKRAAC_type14.setHidden(True)
            self.PseKRAAC_type15.setHidden(True)
            self.PseKRAAC_type16.setHidden(True)
            self.protein_desc_slim_button.setText('Show all descriptors')
        else:
            self.AAC.setHidden(False)
            self.DPC.setHidden(False)
            self.TPC.setHidden(False)
            self.binary_6bit.setHidden(False)
            self.binary_5bit_type1.setHidden(False)
            self.binary_5bit_type2.setHidden(False)
            self.binary_3bit_type1.setHidden(False)
            self.binary_3bit_type2.setHidden(False)
            self.binary_3bit_type3.setHidden(False)
            self.binary_3bit_type4.setHidden(False)
            self.binary_3bit_type5.setHidden(False)
            self.binary_3bit_type6.setHidden(False)
            self.binary_3bit_type7.setHidden(False)
            self.GAAC.setHidden(False)
            self.GDPC.setHidden(False)
            self.GTPC.setHidden(False)
            self.KSCTriad.setHidden(False)
            self.OPF_7bit_type1.setHidden(False)
            self.OPF_7bit_type2.setHidden(False)
            self.OPF_7bit_type3.setHidden(False)
            self.proteinAC.setHidden(False)
            self.proteinCC.setHidden(False)
            self.PseKRAAC_type2.setHidden(False)
            self.PseKRAAC_type3A.setHidden(False)
            self.PseKRAAC_type3B.setHidden(False)
            self.PseKRAAC_type4.setHidden(False)
            self.PseKRAAC_type5.setHidden(False)
            self.PseKRAAC_type6A.setHidden(False)
            self.PseKRAAC_type6B.setHidden(False)
            self.PseKRAAC_type6C.setHidden(False)
            self.PseKRAAC_type7.setHidden(False)
            self.PseKRAAC_type8.setHidden(False)
            self.PseKRAAC_type9.setHidden(False)
            self.PseKRAAC_type10.setHidden(False)
            self.PseKRAAC_type11.setHidden(False)
            self.PseKRAAC_type12.setHidden(False)
            self.PseKRAAC_type13.setHidden(False)
            self.PseKRAAC_type14.setHidden(False)
            self.PseKRAAC_type15.setHidden(False)
            self.PseKRAAC_type16.setHidden(False)
            self.protein_desc_slim_button.setText('Show descriptor slims')

    def run_calculating_protein_descriptors(self):
        if self.protein_sequence_file is not None and len(self.protein_selected_descriptors) != 0:
            self.protein_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.calculate_protein_descriptor)
            t.start()
        else:
            QMessageBox.warning(self, 'Warning', 'Please check your input!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        
    def calculate_protein_descriptor(self):
        try:
            self.protein_descriptor = None
            self.protein_encodings = None
            self.protein_descriptor = iSequence.Descriptor(self.protein_sequence_file, self.protein_default_para)
            if self.protein_descriptor.error_msg == '' and self.protein_descriptor.sequence_type == 'Protein' and len(self.protein_selected_descriptors) > 0:
                self.protein_message_signal.emit('Start calculating descriptors ...')                
                self.setDisabled(True)
                for desc in self.protein_selected_descriptors:
                    # copy parameters for each descriptor
                    if desc in self.protein_para_dict:
                        for key in self.protein_para_dict[desc]:
                            self.protein_default_para[key] = self.protein_para_dict[desc][key]
                    
                    self.protein_message_signal.emit('Calculating descriptor {0} ...'.format(desc))
                    descriptor_name = re.sub(' ', '_', desc)
                    cmd = 'self.protein_descriptor.' + self.protein_descriptor.sequence_type + '_' + descriptor_name + '()'
                    status = eval(cmd)
                    # merge codings
                    if status:
                        if self.protein_encodings is None:
                            self.protein_encodings = self.protein_descriptor.encoding_array
                        else:                            
                            self.protein_encodings = np.hstack((self.protein_encodings, self.protein_descriptor.encoding_array[:, 2:]))
                    else:
                        self.protein_message_signal.emit('Descriptor {0} calculate failed.'.format(desc))

                if self.protein_encodings is not None:
                    self.protein_encodings = np.delete(self.protein_encodings, 1, axis=1)                
                self.protein_message_signal.emit('Descriptors calculate complete.')
                self.protein_display_signal.emit()                
            else:
                self.display_warning_signal.emit('Please check your input!')
                self.setDisabled(False)
                self.protein_progress_bar.clear()            
        except Exception as e:
            self.display_error_signal.emit(str(e))
            self.setDisabled(False)
            self.protein_progress_bar.clear()

    def set_protein_table_content(self):
        if self.protein_encodings is not None:
            # display coding values
            self.protein_desc_tableWidget.init_data(self.protein_encodings[0], self.protein_encodings[1:])
                        
            # Draw histogram
            self.protein_desc_hist_layout.removeWidget(self.protein_desc_histogram)
            sip.delete(self.protein_desc_histogram)
            data = self.protein_encodings[1:, 1:].astype(float)
            self.protein_desc_histogram = PlotWidgets.HistogramWidget()
            self.protein_desc_histogram.init_data('All data', data)
            self.protein_desc_hist_layout.addWidget(self.protein_desc_histogram)
            
            # draw heatmap
            data = pd.DataFrame(self.protein_encodings[1:, 1:].astype(float), columns=self.protein_encodings[0, 1:], index=self.protein_encodings[1:, 0])
            self.protein_heatmap_layout.removeWidget(self.protein_heatmap)
            sip.delete(self.protein_heatmap)
            self.protein_heatmap = PlotWidgets.HeatMapSpanSelector()
            self.protein_heatmap.init_data(data, 'Features', 'Samples')
            self.protein_heatmap_layout.addWidget(self.protein_heatmap)

            # draw boxplot
            self.protein_boxplot_layout.removeWidget(self.protein_boxplot)
            sip.delete(self.protein_boxplot)
            self.protein_boxplot = PlotWidgets.BoxplotSpanSelector()            
            self.protein_boxplot.init_data(data, 'Feature names', 'Value')            
            self.protein_boxplot_layout.addWidget(self.protein_boxplot)
            
            # draw relations
            self.protein_relation_layout.removeWidget(self.protein_relation)
            sip.delete(self.protein_relation)
            self.protein_relation = PlotWidgets.CircosWidget()
            self.protein_relation.init_data(data)
            self.protein_relation_layout.addWidget(self.protein_relation)
            
            # other operation            
            self.setDisabled(False)
            self.protein_progress_bar.clear()
        else:
            QMessageBox.critical(self, 'Error', str(self.protein_descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            self.setDisabled(False)
            self.protein_progress_bar.clear()

    """ events in dna widgetUI """
    def dna_display_message(self, message):
        self.dna_status_label.setText(message)

    def recover_dna_treeWidget(self):
        child_num = self.DNA.childCount()
        for i in range(child_num):
            self.DNA.child(i).setCheckState(0, Qt.Unchecked)

    def dna_panel_clear(self):
        self.dna_sequence_file = None
        self.dna_descriptor = None
        self.dna_encodings = None
        self.dna_desc_tableWidget.tableWidget.clear()
        self.dna_file_lineEdit.clear()
        self.DNA.setDisabled(True)
        self.dna_status_label.setText('Welcome to iFeatureOmega.')
        self.recover_dna_treeWidget()
        self.dna_selected_descriptors = set([])

    def check_dna_descriptors(self):
        if self.dna_descriptor is not None and not self.dna_descriptor.is_equal:
            self.ANF.setDisabled(True)
            self.ENAC.setDisabled(True)
            self.DNAbinary.setDisabled(True)
            self.dnaPS2.setDisabled(True)
            self.dnaPS3.setDisabled(True)
            self.dnaPS4.setDisabled(True)
            self.NCP.setDisabled(True)
            # self.PSTNPss.setDisabled(True)
            # self.PSTNPds.setDisabled(True)
            self.EIIP.setDisabled(True)
            self.dnaDBE.setDisabled(True)
            self.dnaLPDF.setDisabled(True)
            self.dnaDPCP2.setDisabled(True)
            self.dnaTPCP2.setDisabled(True)
            # self.dnaKNN.setDisabled(True)

    def get_dna_file_name(self):
        self.dna_panel_clear()
        self.dna_sequence_file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'Plain text (*.*)')
        if ok:
            self.dna_file_lineEdit.setText(self.dna_sequence_file)
            self.dna_descriptor = iSequence.Sequence(self.dna_sequence_file)
            if self.dna_descriptor.error_msg != '':
                QMessageBox.critical(self, 'Error', str(self.dna_descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.dna_panel_clear()
            elif self.dna_descriptor.sequence_type != 'DNA':
                QMessageBox.critical(self, 'Error', 'The input seems not DNA sequence!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.dna_panel_clear()
            else:
                self.DNA.setDisabled(False)
                self.check_dna_descriptors()
        else:
            self.dna_panel_clear()

    def dna_desc_tree_clicked(self, index):
        item = self.dna_desc_treeWidget.currentItem()
        if item and item.text(0) not in ['DNA']:
            if item.checkState(0) == Qt.Unchecked:
                item.setCheckState(0, Qt.Checked)
                self.dna_selected_descriptors.add(item.text(0))
                self.dna_para_setting(item.text(0))
            else:
                item.setCheckState(0, Qt.Unchecked)
                self.dna_selected_descriptors.discard(item.text(0))  

    def dna_para_setting(self, desc):
        if self.dna_descriptor is not None:
            if desc in ['Kmer', 'RCKmer']:
                num, ok = QInputDialog.getInt(self, '%s setting' % desc, 'Kmer size', 3, 1, 6, 1)
                if ok:
                    self.dna_para_dict[desc]['kmer'] = num
            elif desc in ['ENAC']:
                num, ok = QInputDialog.getInt(self, '%s setting' %desc, 'Sliding window size', 5, 2, 10, 1)
                if ok:
                    self.dna_para_dict[desc]['sliding_window'] = num
            elif desc in ['CKSNAP']:
                num, ok = QInputDialog.getInt(self, '%s setting' %desc, 'K-space number', 3, 0, 5, 1)
                if ok:
                    self.dna_para_dict[desc]['kspace'] = num
            elif desc in ['DPCP', 'DPCP type2']:
                property, ok = InputDialog.QDNADPCPInput.getValues()
                if ok:
                    self.dna_para_dict[desc]['Di-DNA-Phychem'] = property
            elif desc in ['TPCP', 'TPCP type2']:
                property, ok = InputDialog.QDNATPCPInput.getValues()
                if ok:
                    self.dna_para_dict[desc]['Tri-DNA-Phychem'] = property
            elif desc in ['DAC', 'DCC', 'DACC']:
                num, property, ok = InputDialog.QDNAACC2Input.getValues(self.dna_descriptor.minimum_length_without_minus - 2)
                if ok:
                    self.dna_para_dict[desc]['nlag'] = num
                    self.dna_para_dict[desc]['Di-DNA-Phychem'] = property
            elif desc in ['TAC', 'TCC', 'TACC']:
                num, property, ok = InputDialog.QDNAACC3Input.getValues(self.dna_descriptor.minimum_length_without_minus - 3)
                if ok:
                    self.dna_para_dict[desc]['nlag'] = num
                    self.dna_para_dict[desc]['Tri-DNA-Phychem'] = property
            elif desc in ['PseDNC', 'PCPseDNC', 'SCPseDNC']:
                num, weight, property, ok = InputDialog.QDNAPse2Input.getValues(self.dna_descriptor.minimum_length_without_minus - 2)
                if ok:
                    self.dna_para_dict[desc]['lambdaValue'] = num
                    self.dna_para_dict[desc]['weight'] = weight
                    self.dna_para_dict[desc]['Di-DNA-Phychem'] = property
            elif desc in ['PCPseTNC', 'SCPseTNC']:
                num, weight, property, ok = InputDialog.QDNAPse3Input.getValues(self.dna_descriptor.minimum_length_without_minus - 3)
                if ok:
                    self.dna_para_dict[desc]['lambdaValue'] = num
                    self.dna_para_dict[desc]['weight'] = weight
                    self.dna_para_dict[desc]['Di-DNA-Phychem'] = property
            elif desc in ['PseKNC']:
                num, weight, kmer, property, ok = InputDialog.QDNAPseKNCInput.getValues(self.dna_descriptor.minimum_length_without_minus - 2)
                if ok:
                    self.dna_para_dict[desc]['lambdaValue'] = num
                    self.dna_para_dict[desc]['weight'] = weight
                    self.dna_para_dict[desc]['kmer'] = kmer
                    self.dna_para_dict[desc]['Di-DNA-Phychem'] = property
            elif desc in ['Mismatch']:
                num_k, num_m, ok = InputDialog.QMismatchInput.getValues()
                if ok:
                    if num_m >= num_k:
                        num_m = num_k - 1
                    self.dna_para_dict[desc]['kmer'] = num_k
                    self.dna_para_dict[desc]['mismatch'] = num_m
            elif desc in ['Subsequence']:
                num, delta, ok = InputDialog.QSubsequenceInput.getValues()
                if ok:
                    self.dna_para_dict[desc]['kmer'] = num
                    self.dna_para_dict[desc]['delta'] = delta
            elif desc in ['DistancePair']:
                num, cp, ok = InputDialog.QDistancePairInput.getValues()
                if ok:
                    self.dna_para_dict[desc]['distance'] = num
                    self.dna_para_dict[desc]['cp'] = cp
            elif desc in ['NMBroto', 'Moran', 'Geary']:
                num, property, ok = InputDialog.QDNAACC2Input.getValues(self.dna_descriptor.minimum_length_without_minus - 2)
                if ok:
                    self.dna_para_dict[desc]['nlag'] = num
                    self.dna_para_dict[desc]['Di-DNA-Phychem'] = property
            else:
                pass

    def show_dna_slims(self):
        if self.dna_desc_slim_button.text() == 'Show descriptor slims':
            self.NAC.setHidden(True)
            self.dnaPS3.setHidden(True)
            self.dnaPS4.setHidden(True)
            self.dnaDPCP2.setHidden(True)
            self.dnaTPCP2.setHidden(True)
            self.dnazcurve12bit.setHidden(True)
            self.dnazcurve36bit.setHidden(True)
            self.dnazcurve48bit.setHidden(True)
            self.dnazcurve144bit.setHidden(True)
            self.DAC.setHidden(True)
            self.DCC.setHidden(True)
            self.TAC.setHidden(True)
            self.TCC.setHidden(True)
            self.dna_desc_slim_button.setText('Show all descriptors')
        else:
            self.NAC.setHidden(False)
            self.dnaPS3.setHidden(False)
            self.dnaPS4.setHidden(False)
            self.dnaDPCP2.setHidden(False)
            self.dnaTPCP2.setHidden(False)
            self.dnazcurve12bit.setHidden(False)
            self.dnazcurve36bit.setHidden(False)
            self.dnazcurve48bit.setHidden(False)
            self.dnazcurve144bit.setHidden(False)
            self.DAC.setHidden(False)
            self.DCC.setHidden(False)
            self.TAC.setHidden(False)
            self.TCC.setHidden(False)
            self.dna_desc_slim_button.setText('Show descriptor slims')

    def run_calculating_dna_descriptors(self):
        if self.dna_sequence_file is not None and len(self.dna_selected_descriptors) != 0:
            self.dna_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.calculate_dna_descriptor)
            t.start()            
        else:
            QMessageBox.warning(self, 'Warning', 'Please check your input!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
  
    def calculate_dna_descriptor(self):
        try:
            self.dna_descriptor = None
            self.dna_encodings = None
            self.dna_descriptor = iSequence.Descriptor(self.dna_sequence_file, self.dna_default_para)
            if self.dna_descriptor.error_msg == '' and self.dna_descriptor.sequence_type == 'DNA' and len(self.dna_selected_descriptors) > 0:
                self.dna_message_signal.emit('Start calculating descriptors ...')                
                self.setDisabled(True)
                for desc in self.dna_selected_descriptors:
                    # copy parameters for each descriptor
                    if desc in self.dna_para_dict:
                        for key in self.dna_para_dict[desc]:
                            self.dna_default_para[key] = self.dna_para_dict[desc][key]                
                    self.dna_message_signal.emit('Calculating descriptor {0} ...'.format(desc))                    
                    if desc in ['DAC', 'TAC']:
                        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, 'DNA', self.dna_default_para)
                        status = self.dna_descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer, desc)
                    elif desc in ['DCC', 'TCC']:
                        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, 'DNA', self.dna_default_para)
                        status = self.dna_descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer, desc)
                    elif desc in ['DACC', 'TACC']:
                        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, 'DNA', self.dna_default_para)
                        status = self.dna_descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer, desc)
                    elif desc in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
                        my_property_name, my_property_value, ok = CheckAccPseParameter.check_Pse_arguments(desc, 'DNA', self.dna_default_para)
                        cmd = 'self.dna_descriptor.' + desc + '(my_property_name, my_property_value)'
                        status = eval(cmd)
                    else:
                        descriptor_name = re.sub(' ', '_', desc)
                        cmd = 'self.dna_descriptor.' + descriptor_name + '()'
                        status = eval(cmd)               
                    # merge codings
                    if status:
                        if self.dna_encodings is None:
                            self.dna_encodings = self.dna_descriptor.encoding_array
                        else:
                            self.dna_encodings = np.hstack((self.dna_encodings, self.dna_descriptor.encoding_array[:, 2:]))
                    else:
                        self.dna_message_signal.emit('Descriptor {0} calculate failed.'.format(desc))
                        self.setDisabled(False)
                        self.dna_progress_bar.clear()
                        
                    if self.dna_encodings is not None:
                        self.dna_encodings = np.delete(self.dna_encodings, 1, axis=1)
                self.dna_message_signal.emit('Descriptors calculate complete.')
                self.dna_display_signal.emit()
            else:
                self.display_warning_signal.emit('Please check your input!')
        except Exception as e:
            self.display_error_signal.emit(str(e))
            self.setDisabled(False)
            self.dna_progress_bar.clear()

    def set_dna_table_content(self):
        if self.dna_encodings is not None:
            self.dna_desc_tableWidget.init_data(self.dna_encodings[0], self.dna_encodings[1:])
                        
            # Draw histogram
            self.dna_desc_hist_layout.removeWidget(self.dna_desc_histogram)
            sip.delete(self.dna_desc_histogram)
            data = self.dna_encodings[1:, 1:].astype(float)
            self.dna_desc_histogram = PlotWidgets.HistogramWidget()
            self.dna_desc_histogram.init_data('All data', data)
            self.dna_desc_hist_layout.addWidget(self.dna_desc_histogram)

            # draw heatmap
            data = pd.DataFrame(self.dna_encodings[1:, 1:].astype(float), columns=self.dna_encodings[0, 1:], index=self.dna_encodings[1:, 0])
            self.dna_heatmap_layout.removeWidget(self.dna_heatmap)
            sip.delete(self.dna_heatmap)
            self.dna_heatmap = PlotWidgets.HeatMapSpanSelector()
            self.dna_heatmap.init_data(data, 'Features', 'Samples')
            self.dna_heatmap_layout.addWidget(self.dna_heatmap)

            # draw boxplot
            self.dna_boxplot_layout.removeWidget(self.dna_boxplot)
            sip.delete(self.dna_boxplot)
            self.dna_boxplot = PlotWidgets.BoxplotSpanSelector()            
            self.dna_boxplot.init_data(data, 'Feature names', 'Value')            
            self.dna_boxplot_layout.addWidget(self.dna_boxplot)
            
            # draw relations
            self.dna_relation_layout.removeWidget(self.dna_relation)
            sip.delete(self.dna_relation)
            self.dna_relation = PlotWidgets.CircosWidget()
            self.dna_relation.init_data(data)
            self.dna_relation_layout.addWidget(self.dna_relation)

            # other operation
            self.setDisabled(False)
            self.dna_progress_bar.clear()
        else:
            QMessageBox.critical(self, 'Error', str(self.dna_descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)            
            self.setDisabled(False)
            self.dna_progress_bar.clear()

    """ events in rna widgetUI """
    def rna_display_message(self, message):
        self.rna_status_label.setText(message)

    def recover_rna_treeWidget(self):
        child_num = self.RNA.childCount()
        for i in range(child_num):
            self.RNA.child(i).setCheckState(0, Qt.Unchecked)
    
    def rna_panel_clear(self):
        self.rna_sequence_file = None
        self.rna_descriptor = None
        self.rna_encodings = None
        self.rna_desc_tableWidget.tableWidget.clear()
        self.rna_file_lineEdit.clear()
        self.RNA.setDisabled(True)
        self.rna_status_label.setText('Welcome to iFeatureOmega.')
        self.recover_rna_treeWidget()
        self.rna_selected_descriptors = set([])

    def check_rna_descriptors(self):
        if self.rna_descriptor is not None and not self.rna_descriptor.is_equal:
            self.RNAENAC.setDisabled(True)
            self.RNAANF.setDisabled(True)
            self.RNANCP.setDisabled(True)
            self.RNAPSTNPss.setDisabled(True)
            self.RNAbinary.setDisabled(True)
            self.rnaPS2.setDisabled(True)
            self.rnaPS3.setDisabled(True)
            self.rnaPS4.setDisabled(True)
            self.rnaDBE.setDisabled(True)
            self.rnaLPDF.setDisabled(True)
            self.rnaDPCP2.setDisabled(True)
            self.rnaKNN.setDisabled(True)

    def get_rna_file_name(self):
        self.rna_panel_clear()
        self.rna_sequence_file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'Plain text (*.*)')
        if ok:
            self.rna_file_lineEdit.setText(self.rna_sequence_file)
            self.rna_descriptor = iSequence.Sequence(self.rna_sequence_file)
            if self.rna_descriptor.error_msg != '':
                QMessageBox.critical(self, 'Error', str(self.rna_descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.rna_panel_clear()
            elif self.rna_descriptor.sequence_type != 'RNA':
                QMessageBox.critical(self, 'Error', 'The input seems not RNA sequence!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.rna_panel_clear()
            else:
                self.RNA.setDisabled(False)
                self.check_rna_descriptors()
        else:
            self.rna_panel_clear()

    def rna_desc_tree_clicked(self, index):
        item = self.rna_desc_treeWidget.currentItem()
        if item and item.text(0) not in ['RNA']:
            if item.checkState(0) == Qt.Unchecked:
                item.setCheckState(0, Qt.Checked)
                self.rna_selected_descriptors.add(item.text(0))
                self.rna_para_setting(item.text(0))
            else:
                item.setCheckState(0, Qt.Unchecked)
                self.rna_selected_descriptors.discard(item.text(0))        

    def rna_para_setting(self, desc):
        if self.rna_descriptor is not None:
            if desc in ['Kmer', 'RCKmer']:
                num, ok = QInputDialog.getInt(self, '%s setting' % desc, 'Kmer size', 3, 1, 6, 1)
                if ok:
                    self.rna_para_dict[desc]['kmer'] = num
            elif desc in ['ENAC']:
                num, ok = QInputDialog.getInt(self, '%s setting' %desc, 'Sliding window size', 5, 2, 10, 1)
                if ok:
                    self.rna_para_dict[desc]['sliding_window'] = num
            elif desc in ['CKSNAP']:
                num, ok = QInputDialog.getInt(self, '%s setting' %desc, 'K-space number', 3, 0, 5, 1)
                if ok:
                    self.rna_para_dict[desc]['kspace'] = num
            elif desc in ['DPCP', 'DPCP type2']:
                property, ok = InputDialog.QRNADPCPInput.getValues()
                if ok:
                    self.rna_para_dict[desc]['Di-RNA-Phychem'] = property                
            elif desc in ['DAC', 'DCC', 'DACC']:
                num, property, ok = InputDialog.QRNAACC2Input.getValues(self.rna_descriptor.minimum_length_without_minus - 2)
                if ok:
                    self.rna_para_dict[desc]['nlag'] = num
                    self.rna_para_dict[desc]['Di-RNA-Phychem'] = property
            elif desc in ['PseDNC', 'PCPseDNC', 'SCPseDNC']:
                num, weight, property, ok = InputDialog.QRNAPse2Input.getValues(self.rna_descriptor.minimum_length_without_minus - 2)
                if ok:
                    self.rna_para_dict[desc]['lambdaValue'] = num
                    self.rna_para_dict[desc]['weight'] = weight
                    self.rna_para_dict[desc]['Di-RNA-Phychem'] = property                
            elif desc in ['PseKNC']:
                num, weight, kmer, property, ok = InputDialog.QRNAPseKNCInput.getValues(self.rna_descriptor.minimum_length_without_minus - 2)
                if ok:
                    self.rna_para_dict[desc]['lambdaValue'] = num
                    self.rna_para_dict[desc]['weight'] = weight
                    self.rna_para_dict[desc]['kmer'] = kmer
                    self.rna_para_dict[desc]['Di-RNA-Phychem'] = property                
            elif desc in ['Mismatch']:
                num_k, num_m, ok = InputDialog.QMismatchInput.getValues()
                if ok:
                    if num_m >= num_k:
                        num_m = num_k - 1
                    self.rna_para_dict[desc]['kmer'] = num_k
                    self.rna_para_dict[desc]['mismatch'] = num_m
            elif desc in ['Subsequence']:
                num, delta, ok = InputDialog.QSubsequenceInput.getValues()
                if ok:
                    self.rna_para_dict[desc]['kmer'] = num
                    self.rna_para_dict[desc]['delta'] = delta
            elif desc in ['DistancePair']:
                num, cp, ok = InputDialog.QDistancePairInput.getValues()
                if ok:
                    self.rna_para_dict[desc]['distance'] = num
                    self.rna_para_dict[desc]['cp'] = cp
            elif desc in ['NMBroto', 'Moran', 'Geary']:
                num, property, ok = InputDialog.QRNAACC2Input.getValues(self.rna_descriptor.minimum_length_without_minus - 2)
                if ok:
                    self.rna_para_dict[desc]['nlag'] = num
                    self.rna_para_dict[desc]['Di-RNA-Phychem'] = property
            else:
                pass

    def show_rna_slims(self):
        if self.rna_desc_slim_button.text() == 'Show descriptor slims':
            self.RNANAC.setHidden(True)
            self.rnaPS3.setHidden(True)
            self.rnaPS4.setHidden(True)
            self.rnaDPCP2.setHidden(True)
            self.rnazcurve12bit.setHidden(True)
            self.rnazcurve36bit.setHidden(True)
            self.rnazcurve48bit.setHidden(True)
            self.rnazcurve144bit.setHidden(True)
            self.RNADAC.setHidden(True)
            self.RNADCC.setHidden(True)
            self.rna_desc_slim_button.setText('Show all descriptors')
        else:
            self.RNANAC.setHidden(False)
            self.rnaPS3.setHidden(False)
            self.rnaPS4.setHidden(False)
            self.rnaDPCP2.setHidden(False)
            self.rnazcurve12bit.setHidden(False)
            self.rnazcurve36bit.setHidden(False)
            self.rnazcurve48bit.setHidden(False)
            self.rnazcurve144bit.setHidden(False)
            self.RNADAC.setHidden(False)
            self.RNADCC.setHidden(False)
            self.rna_desc_slim_button.setText('Show descriptor slims')

    def run_calculating_rna_descriptors(self):
        if self.rna_sequence_file is not None and len(self.rna_selected_descriptors) != 0:
            self.rna_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.calculate_rna_descriptor)
            t.start()
        else:
            QMessageBox.warning(self, 'Warning', 'Please check your input!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
  
    def calculate_rna_descriptor(self):
        try:
            self.rna_descriptor = None
            self.rna_encodings = None
            self.rna_descriptor = iSequence.Descriptor(self.rna_sequence_file, self.rna_default_para)
            if self.rna_descriptor.error_msg == '' and self.rna_descriptor.sequence_type == 'RNA' and len(self.rna_selected_descriptors) > 0:
                self.rna_message_signal.emit('Start calculating descriptors ...')
                self.setDisabled(True)
                for desc in self.rna_selected_descriptors:
                    # copy parameters for each descriptor
                    if desc in self.rna_para_dict:
                        for key in self.rna_para_dict[desc]:
                            self.rna_default_para[key] = self.rna_para_dict[desc][key]                
                    self.rna_message_signal.emit('Calculating descriptor {0} ...'.format(desc))                    
                    if desc in ['DAC', 'TAC']:
                        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, 'RNA', self.rna_default_para)
                        status = self.rna_descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer, desc)
                    elif desc in ['DCC', 'TCC']:
                        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, 'RNA', self.rna_default_para)
                        status = self.rna_descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer, desc)
                    elif desc in ['DACC', 'TACC']:
                        my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, 'RNA', self.rna_default_para)
                        status = self.rna_descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer, desc)
                    elif desc in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
                        my_property_name, my_property_value, ok = CheckAccPseParameter.check_Pse_arguments(desc, 'RNA', self.rna_default_para)
                        cmd = 'self.rna_descriptor.' + desc + '(my_property_name, my_property_value)'
                        status = eval(cmd)
                    else:
                        descriptor_name = re.sub(' ', '_', desc)
                        cmd = 'self.rna_descriptor.' + descriptor_name + '()'
                        status = eval(cmd)
                    # merge codings
                    if status:
                        if self.rna_encodings is None:
                            self.rna_encodings = self.rna_descriptor.encoding_array
                        else:
                            self.rna_encodings = np.hstack((self.rna_encodings, self.rna_descriptor.encoding_array[:, 2:]))
                    else:
                        self.rna_message_signal.emit('Descriptor {0} calculate failed.'.format(desc))
                        self.setDisabled(False)
                        self.rna_progress_bar.clear()
                if self.rna_encodings is not None:
                        self.rna_encodings = np.delete(self.rna_encodings, 1, axis=1)
                self.rna_message_signal.emit('Descriptors calculate complete.')
                self.rna_display_signal.emit()
            else:
                self.display_warning_signal('Please check your input!')
        except Exception as e:
            self.display_error_signal.emit(str(e))
            self.setDisabled(False)
            self.rna_progress_bar.clear()

    def set_rna_table_content(self):
        if self.rna_encodings is not None:
            self.rna_desc_tableWidget.init_data(self.rna_encodings[0], self.rna_encodings[1:])
                        
            # Draw histogram
            self.rna_desc_hist_layout.removeWidget(self.rna_desc_histogram)
            sip.delete(self.rna_desc_histogram)
            data = self.rna_encodings[1:, 1:].astype(float)
            self.rna_desc_histogram = PlotWidgets.HistogramWidget()
            self.rna_desc_histogram.init_data('All data', data)
            self.rna_desc_hist_layout.addWidget(self.rna_desc_histogram)

            # draw heatmap
            data = pd.DataFrame(self.rna_encodings[1:, 1:].astype(float), columns=self.rna_encodings[0, 1:], index=self.rna_encodings[1:, 0])            
            self.rna_heatmap_layout.removeWidget(self.rna_heatmap)
            sip.delete(self.rna_heatmap)
            self.rna_heatmap = PlotWidgets.HeatMapSpanSelector()
            self.rna_heatmap.init_data(data, 'Features', 'Samples')
            self.rna_heatmap_layout.addWidget(self.rna_heatmap)

            # draw boxplot
            self.rna_boxplot_layout.removeWidget(self.rna_boxplot)
            sip.delete(self.rna_boxplot)
            self.rna_boxplot = PlotWidgets.BoxplotSpanSelector()            
            self.rna_boxplot.init_data(data, 'Feature names', 'Value')            
            self.rna_boxplot_layout.addWidget(self.rna_boxplot)

            # draw relations
            self.rna_relation_layout.removeWidget(self.rna_relation)
            sip.delete(self.rna_relation)
            self.rna_relation = PlotWidgets.CircosWidget()
            self.rna_relation.init_data(data)
            self.rna_relation_layout.addWidget(self.rna_relation)

            # other operation
            self.setDisabled(False)
            self.rna_progress_bar.clear()
        else:
            QMessageBox.critical(self, 'Error', str(self.rna_descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            # self.rna_widget.setDisabled(False)
            self.setDisabled(False)
            self.rna_progress_bar.clear()

    """ events in structure widgetUI """
    def structure_display_message(self, message):
        self.structure_status_label.setText(message)

    def recover_structure_treeWidget(self):
        child_num = self.Structure.childCount()
        for i in range(child_num):
            self.Structure.child(i).setCheckState(0, Qt.Unchecked)
    
    def structure_panel_clear(self):
        self.structure_sequence_file = None
        self.structure_descriptor = None
        self.structure_encodings = None
        self.structure_desc_tableWidget.tableWidget.clear()
        self.structure_file_lineEdit.clear()
        # self.pdb_accession_lineEdit.clear()
        self.Structure.setDisabled(True)
        self.structure_status_label.setText('Welcome to iFeatureOmega.')
        # self.recover_structure_treeWidget()
        self.structure_selected_descriptors = set([])

    def upload_pdb_file(self):
        self.structure_panel_clear()
        self.structure_file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'PDB file (*.pdb);;CIF file(*.cif)')
        if ok:
            self.structure_file_lineEdit.setText(self.structure_file)
            self.structure_descriptor = iStructure.Structure(self.structure_file)
            if self.structure_descriptor.error_msg != '' and self.structure_descriptor.error_msg is not None:
                QMessageBox.critical(self, 'Error', str(self.structure_descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.structure_panel_clear()
            else:
                self.Structure.setDisabled(False)
        else:
            self.structure_panel_clear()
            
    def download_pdb_file(self):
        self.structure_panel_clear()
        pdb_id = self.pdb_accession_lineEdit.text()
        if len(pdb_id) != 4:
            QMessageBox.critical(self, 'Error', 'The input seems not a PDB ID.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            self.setDisabled(True)
            self.structure_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=lambda: self.download(pdb_id))
            t.start()

    def download(self, id):
        self.structure_message_signal.emit('Downloading structure file for ID {0} ...'.format(id))
        tmp = tempfile.gettempdir()
        from Bio.PDB.PDBList import PDBList
        pdb1 = PDBList()
        self.structure_file = pdb1.retrieve_pdb_file(id, pdir=tmp)
        if os.path.exists(self.structure_file):
            self.structure_message_signal.emit('File download successfully.')
            # self.structure_file_lineEdit.setText(self.structure_file)
        else:
            self.structure_message_signal.emit('File download failed.')
        self.setDisabled(False)
        self.structure_progress_bar.clear()
        
        self.structure_descriptor = iStructure.Structure(self.structure_file)
        if self.structure_descriptor.error_msg != '' and self.structure_descriptor.error_msg is not None:
            QMessageBox.critical(self, 'Error', str(self.structure_descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            self.structure_panel_clear()
        else:
            self.Structure.setDisabled(False)

    def structure_desc_tree_clicked(self, index):
        item = self.structure_desc_treeWidget.currentItem()
        if item and item.text(0) not in ['Protein structure']:
            self.structure_selected_descriptors = item.text(0)
            if item.text(0) in ['AAC_type1', 'AAC_type2', 'GAAC_type1', 'GAAC_type2', 'SS3_type1', 'SS3_type2', 'SS8_type1', 'SS8_type2']:
               shell, ok = InputDialog.QStructureResidueDialog().getValues()
               if ok:
                   self.structure_default_para['residue_shell'] = shell
            if item.text(0) in ['AC_type1', 'AC_type2']:
               shell, ok = InputDialog.QStructureAtomDialog().getValues()
               if ok:
                   self.structure_default_para['atom_shell'] = shell
                
    def run_calculating_structure_descriptors(self):
        if self.structure_descriptor is not None:
            self.structure_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.calculate_structure_descriptor)
            t.start()
        else:
            QMessageBox.warning(self, 'Warning', 'Please check your input!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
  
    def calculate_structure_descriptor(self):
        try:
            if self.structure_descriptor.read_pdb():
                self.structure_message_signal.emit('Start calculating descriptors ...')                
                self.setDisabled(True)
                if self.structure_selected_descriptors == 'AAC_type1':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_residue_descriptor([], method='AAC_type1', shell=self.structure_default_para['residue_shell'])
                elif self.structure_selected_descriptors == 'AAC_type2':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_residue_descriptor([], method='AAC_type2', shell=self.structure_default_para['residue_shell'])
                elif self.structure_selected_descriptors == 'GAAC_type1':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_residue_descriptor([], method='GAAC_type1', shell=self.structure_default_para['residue_shell'])
                elif self.structure_selected_descriptors == 'GAAC_type2':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_residue_descriptor([], method='GAAC_type2', shell=self.structure_default_para['residue_shell'])
                elif self.structure_selected_descriptors == 'SS3_type1':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_residue_descriptor([], method='SS3_type1', shell=self.structure_default_para['residue_shell'])
                elif self.structure_selected_descriptors == 'SS3_type2':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_residue_descriptor([], method='SS3_type2', shell=self.structure_default_para['residue_shell'])
                elif self.structure_selected_descriptors == 'SS8_type1':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_residue_descriptor([], method='SS8_type1', shell=self.structure_default_para['residue_shell'])
                elif self.structure_selected_descriptors == 'SS8_type2':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_residue_descriptor([], method='SS8_type2', shell=self.structure_default_para['residue_shell'])
                elif self.structure_selected_descriptors == 'AC_type1':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_atom_descriptor([], method='AC_type1', shell=self.structure_default_para['atom_shell'])
                elif self.structure_selected_descriptors == 'AC_type2':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_atom_descriptor([], method='AC_type2', shell=self.structure_default_para['atom_shell'])
                elif self.structure_selected_descriptors == 'Network-based index':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_network_descriptor([])
                elif self.structure_selected_descriptors == 'HSE_CA':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_HSE_CA()
                elif self.structure_selected_descriptors == 'HSE_CB':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_HSE_CB()
                elif self.structure_selected_descriptors == 'Residue depth':
                    status, self.structure_descriptor.encodings = self.structure_descriptor.get_residue_depth()            
                if status:
                    self.structure_encodings = self.structure_descriptor.encodings
                    self.structure_message_signal.emit('Descriptor calculate complete.')
                    self.structure_display_signal.emit()
                else:
                    self.structure_message_signal.emit('Descriptor calculate failed.')
                    self.display_error_signal.emit(str(self.structure_descriptor.error_msg))                   
                    self.setDisabled(False)
                    self.structure_progress_bar.clear()
            else:
                self.display_error_signal.emit('Calculate failed, please check your input!')
                self.setDisabled(False)
                self.structure_progress_bar.clear()
        except Exception as e:
            self.display_error_signal.emit(str(e))
            self.setDisabled(False)
            self.structure_progress_bar.clear()

    def set_structure_table_content(self):
        if self.structure_descriptor.encodings is not None:
            self.structure_desc_tableWidget.init_data(self.structure_descriptor.encodings.columns, self.structure_descriptor.encodings.values)            

            # Draw histogram
            self.structure_desc_hist_layout.removeWidget(self.structure_desc_histogram)
            sip.delete(self.structure_desc_histogram)
            data = self.structure_descriptor.encodings.values[:, 1:].astype(float)
            self.structure_desc_histogram = PlotWidgets.HistogramWidget()
            self.structure_desc_histogram.init_data('All data', data)
            self.structure_desc_hist_layout.addWidget(self.structure_desc_histogram)

            # draw heatmap
            data = pd.DataFrame(self.structure_descriptor.encodings.values[:, 1:].astype(float), columns=self.structure_descriptor.encodings.columns[1:], index=self.structure_descriptor.encodings.values[:, 0])
            self.structure_heatmap_layout.removeWidget(self.structure_heatmap)
            sip.delete(self.structure_heatmap)
            self.structure_heatmap = PlotWidgets.HeatMapSpanSelector()
            self.structure_heatmap.init_data(data, 'Features', 'Samples')
            self.structure_heatmap_layout.addWidget(self.structure_heatmap)

            # draw boxplot
            self.structure_boxplot_layout.removeWidget(self.structure_boxplot)
            sip.delete(self.structure_boxplot)
            self.structure_boxplot = PlotWidgets.BoxplotSpanSelector()            
            self.structure_boxplot.init_data(data, 'Feature names', 'Value')            
            self.structure_boxplot_layout.addWidget(self.structure_boxplot)

            # draw relations
            self.structure_relation_layout.removeWidget(self.structure_relation)
            sip.delete(self.structure_relation)
            self.structure_relation = PlotWidgets.CircosWidget()
            self.structure_relation.init_data(data)
            self.structure_relation_layout.addWidget(self.structure_relation)

            # other operation
            self.setDisabled(False)
            self.structure_progress_bar.clear()
        else:
            QMessageBox.critical(self, 'Error', str(self.structure_descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            self.setDisabled(False)
            self.structure_progress_bar.clear()

    """ events in chemical widgetUI """       
    def chemical_display_message(self, message):
        self.chemical_status_label.setText(message)

    def recover_chemical_treeWidget(self):
        child_num = self.Chemical.childCount()
        for i in range(child_num):
            self.Chemical.child(i).setCheckState(0, Qt.Unchecked)
    
    def chemical_panel_clear(self):
        self.chemical_smiles_file = None        
        self.chemical_encodings = None
        self.chemical_desc_tableWidget.tableWidget.clear()
        self.chemical_file_lineEdit.clear()  
        self.recover_chemical_treeWidget()      
        self.Chemical.setDisabled(True)
        self.chemical_status_label.setText('Welcome to iFeatureOmega.')        
        self.chemical_selected_descriptors = set([])

    def get_smiles_file(self):
        self.chemical_panel_clear()
        self.chemical_smiles_file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'Plain text (*.*)')
        if ok:
            self.chemical_file_lineEdit.setText(self.chemical_smiles_file)
            self.Chemical.setDisabled(False)
        else:
            self.chemical_panel_clear()

    def chemical_desc_tree_clicked(self, index):
        item = self.chemical_desc_treeWidget.currentItem()
        if item and item.text(0) not in ['Chemical']:
            if item.checkState(0) == Qt.Unchecked:
                item.setCheckState(0, Qt.Checked)
                self.chemical_selected_descriptors.add(item.text(0))
            else:
                item.setCheckState(0, Qt.Unchecked)
                self.chemical_selected_descriptors.discard(item.text(0))
            
    def run_calculating_chemical_descriptors(self):
        if self.chemical_smiles_file is not None and os.path.exists(self.chemical_smiles_file) and len(self.chemical_selected_descriptors) > 0:
            self.chemical_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.calculate_chemical_descriptor)
            t.start()            
        else:
            QMessageBox.warning(self, 'Warning', 'Please check your input!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        
    def calculate_chemical_descriptor(self):
        # try:
        self.setDisabled(True)
        self.chemical_encodings = None
        with open(self.chemical_smiles_file) as f:
            smiles_list = f.read().strip().split('\n')
        
        mol_list = []
        no_error = True
        for mol in smiles_list:
            molObj = Chem.MolFromSmiles(mol)
            if molObj is not None:
                mol_list.append(molObj)
            else:
                no_error = False
        if no_error:
            fps = []
            for desc in self.chemical_selected_descriptors:
                if desc in self.chemical_default_parameters:
                    fps += self.chemical_default_parameters[desc]
            ligand = iChemical.Ligand(fps=fps)
            df = ligand(mol_list)
            if len(df) == len(smiles_list):
                df.insert(0, 'SMILES', np.array(smiles_list))                    
                self.chemical_encodings = df
                self.chemical_message_signal.emit('Descriptor calculate complete.')
                self.chemical_display_signal.emit()
            else:
                self.display_error_signal.emit('Descriptor calculate failed.')                    
                self.chemical_panel_clear()                    
                self.setDisabled(False)
                self.chemical_progress_bar.clear()
        else:
            self.display_warning_signal.emit('Failed parsing SMILES.')                
            self.chemical_panel_clear()
            self.setDisabled(False)
            self.chemical_progress_bar.clear()
        # except Exception as e:
        #     self.display_error_signal.emit(str(e))            
        #     self.setDisabled(False)
        #     self.chemical_progress_bar.clear()

    def set_chemical_table_content(self):
        if self.chemical_encodings is not None:
            self.chemical_desc_tableWidget.init_data(self.chemical_encodings.columns, self.chemical_encodings.values)
                        
            # Draw histogram
            self.chemical_desc_hist_layout.removeWidget(self.chemical_desc_histogram)
            sip.delete(self.chemical_desc_histogram)
            data = self.chemical_encodings.values[:, 1:].astype(float)
            self.chemical_desc_histogram = PlotWidgets.HistogramWidget()
            self.chemical_desc_histogram.init_data('All data', data)
            self.chemical_desc_hist_layout.addWidget(self.chemical_desc_histogram)           

            # draw heatmap
            data = pd.DataFrame(self.chemical_encodings.values[:, 1:].astype(float), columns=self.chemical_encodings.columns[1:], index=self.chemical_encodings.values[:, 0])
            self.chemical_heatmap_layout.removeWidget(self.chemical_heatmap)
            sip.delete(self.chemical_heatmap)
            self.chemical_heatmap = PlotWidgets.HeatMapSpanSelector()
            self.chemical_heatmap.init_data(data, 'Features', 'Samples')
            self.chemical_heatmap_layout.addWidget(self.chemical_heatmap)

            # draw boxplot
            self.chemical_boxplot_layout.removeWidget(self.chemical_boxplot)
            sip.delete(self.chemical_boxplot)
            self.chemical_boxplot = PlotWidgets.BoxplotSpanSelector()            
            self.chemical_boxplot.init_data(data, 'Feature names', 'Value')            
            self.chemical_boxplot_layout.addWidget(self.chemical_boxplot)

            # draw relations
            self.chemical_relation_layout.removeWidget(self.chemical_relation)
            sip.delete(self.chemical_relation)
            self.chemical_relation = PlotWidgets.CircosWidget()
            self.chemical_relation.init_data(data)
            self.chemical_relation_layout.addWidget(self.chemical_relation)

            # other operation
            # self.chemical_widget.setDisabled(False)
            self.setDisabled(False)
            self.chemical_progress_bar.clear()
        else:
            QMessageBox.critical(self, 'Error', 'Descriptor calculate failed.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            # self.chemical_widget.setDisabled(False)
            self.setDisabled(False)
            self.chemical_progress_bar.clear()
    
    """ events in analysis widgetUI """
    def analysis_display_message(self, message):
        self.analysis_status_label.setText(message)
    
    def analysis_panel_clear(self):
        self.analysis_data_file = None
        self.analysis_data = None
        self.analysis_file_lineEdit.setText('')
        self.analysis_file_lineEdit2.setText('')
        self.analysis_data_lineEdit.setText('')        
        
    def data_from_file(self):
        self.analysis_data_file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        if ok:
            self.analysis_data = DataAnalysis.ILearnData(self.analysis_default_para)
            ok1 = self.analysis_data.load_data_from_file(self.analysis_data_file)
            if ok1:
                self.analysis_datashape.setText('Data shape: (%s, %s)' %(self.analysis_data.row, self.analysis_data.column))
                self.analysis_file_lineEdit.setText(self.analysis_data_file)
                self.analysis_data_lineEdit.setText('')
            else:
                self.analysis_panel_clear()
                QMessageBox.critical(self, 'Error', str(self.analysis_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            self.analysis_panel_clear()

    def data_from_file2(self):
        self.analysis_data_file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        if ok:
            self.analysis_data = DataAnalysis.ILearnData(self.analysis_default_para)
            ok1 = self.analysis_data.load_data_from_file2(self.analysis_data_file)
            if ok1:
                self.analysis_datashape.setText('Data shape: (%s, %s)' %(self.analysis_data.row, self.analysis_data.column))
                self.analysis_file_lineEdit2.setText(self.analysis_data_file)
                self.analysis_file_lineEdit.setText('')
                self.analysis_data_lineEdit.setText('')
            else:
                self.analysis_panel_clear()
                QMessageBox.critical(self, 'Error', str(self.analysis_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            self.analysis_panel_clear()

    def data_from_panels(self):
        data_source, ok = InputDialog.QDataSelection2.getValues(Protein=self.protein_encodings, DNA=self.dna_encodings, RNA=self.rna_encodings, Structure=self.structure_encodings, Chemical=self.chemical_encodings)
        if ok:
            self.analysis_data = DataAnalysis.ILearnData(self.analysis_default_para)
            if data_source in ['Protein descriptor data']:
                ok1 = self.analysis_data.load_data_from_descriptor(self.protein_encodings)
            if data_source in ['DNA descriptor data']:
                ok1 = self.analysis_data.load_data_from_descriptor(self.dna_encodings)
            if data_source in ['RNA descriptor data']:
                ok1 = self.analysis_data.load_data_from_descriptor(self.rna_encodings)
            if data_source in ['Structure descriptor data']:
                ok1 = self.analysis_data.load_data_from_df(self.structure_encodings)
            if data_source in ['Chemical descriptor data']:
                ok1 = self.analysis_data.load_data_from_df(self.chemical_encodings)
            
            if ok1:
                self.analysis_data_lineEdit.setText(data_source)
                self.analysis_datashape.setText('Data shape: (%s, %s)' %(self.analysis_data.row, self.analysis_data.column))
                self.analysis_file_lineEdit.setText('')
                self.analysis_file_lineEdit2.setText('')
            else:
                self.analysis_panel_clear()
                QMessageBox.critical(self, 'Error', str(self.analysis_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)        
        else:
            self.analysis_panel_clear()
                
    def analysis_tree_clicked(self, index):
        item = self.analysis_treeWidget.currentItem()
        if item and item.text(0) not in ['Cluster algorithms', 'Dimensionality reduction algorithms', 'Feature normalization algorithms']:
            self.analysis_type = item.parent().text(0)
            self.analysis_analysisType_lineEdit.setText(self.analysis_type)
            if item.text(0) in ['kmeans', 'MiniBatchKMeans', 'GM', 'Agglomerative', 'Spectral']:
                self.analysis_selected_algorithm = item.text(0)
                self.analysis_algorithm_lineEdit.setText(self.analysis_selected_algorithm)
                num, ok = QInputDialog.getInt(self, '%s setting' % self.analysis_selected_algorithm, 'Cluster number', 2, 2, 10, 1)
                if ok:
                    self.analysis_default_para['nclusters'] = num
                    self.analysis_para_lineEdit.setText('Cluster number: %s' % num)
            elif item.text(0) in ['MCL']:
                self.analysis_selected_algorithm = item.text(0)
                self.analysis_algorithm_lineEdit.setText(self.analysis_selected_algorithm)
                expand, inflate, mult, ok = InputDialog.QMCLInput.getValues()
                if ok:
                    self.analysis_default_para['expand_factor'] = expand
                    self.analysis_default_para['inflate_factor'] = inflate
                    self.analysis_default_para['multiply_factor'] = mult
                    self.analysis_para_lineEdit.setText('Expand: %s; Inflate: %s; Multiply: %s' % (expand, inflate, mult))
            elif item.text(0) in ['hcluster', 'APC', 'meanshift', 'DBSCAN']:
                self.analysis_selected_algorithm = item.text(0)
                self.analysis_algorithm_lineEdit.setText(self.analysis_selected_algorithm)
                self.analysis_para_lineEdit.setText('None')
            elif item.text(0) in ['PCA', 't_SNE', 'LDA']:
                self.analysis_selected_algorithm = item.text(0)
                self.analysis_algorithm_lineEdit.setText(self.analysis_selected_algorithm)
                num, ok = QInputDialog.getInt(self, '%s setting' % self.analysis_selected_algorithm, 'Reduced number of dimensions', 2, 2, 10000, 1)
                if ok:
                    self.analysis_default_para['n_components'] = num                    
                    self.analysis_para_lineEdit.setText('Reduced number of dimensions: %s' % num)
            else:
                self.analysis_selected_algorithm = item.text(0)
                self.analysis_algorithm_lineEdit.setText(self.analysis_selected_algorithm)
                self.analysis_para_lineEdit.setText('None')
            
    def run_data_analysis(self):
        if self.analysis_selected_algorithm is not None and self.analysis_data is not None:
            self.analysis_status = False
            self.analysis_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.data_analysis)
            t.start()
        else:
            if self.analysis_data is None:
                QMessageBox.critical(self, 'Error', 'Empty data!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            else:
                QMessageBox.critical(self, 'Error', 'Please select an analysis algorithm.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
    
    def data_analysis(self):
        try:
            if self.analysis_selected_algorithm is not None and self.analysis_data is not None:
                # self.analysis_widget.setDisabled(True)
                self.setDisabled(True)
                self.analysis_status_label.setText('Calculating ...')
                if self.analysis_type == 'Cluster algorithms':
                    cmd = 'self.analysis_data.' + self.analysis_selected_algorithm + '()'
                    try:
                        status = eval(cmd)
                        self.analysis_status = status
                    except Exception as e:
                        self.analysis_data.error_msg = 'Clustering failed.'
                        status = False
                elif self.analysis_type == 'Dimensionality reduction algorithms':
                    if self.analysis_selected_algorithm == 't_SNE':
                        algo = 't_sne'
                    else:
                        algo = self.analysis_selected_algorithm
                    cmd = 'self.analysis_data.' + algo + '(self.analysis_default_para["n_components"])'
                    try:
                        self.analysis_data.dimension_reduction_result, status = eval(cmd)
                        self.analysis_data.cluster_plot_data, _ = self.analysis_data.t_sne(2)
                        self.analysis_status = status
                    except Exception as e:
                        self.analysis_data.error_msg = str(e)
                        self.analysis_status = False
                else:
                    cmd = 'self.analysis_data.' + self.analysis_selected_algorithm + '()'
                    status = False
                    try:
                        status = eval(cmd)
                        self.analysis_status = status
                    except Exception as e:
                        QMessageBox.critical(self, 'Error', 'Calculate failed.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                        self.analysis_data.error_msg = str(e)
                        self.analysis_status = False
                self.analysis_display_signal.emit()            
                self.analysis_progress_bar.clear()
            else:
                self.analysis_progress_bar.clear()
                if self.analysis_data is None:
                    QMessageBox.critical(self, 'Error', 'Empty data!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                elif self.analysis_selected_algorithm == '':
                    QMessageBox.critical(self, 'Error', 'Please select an analysis angorithm', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                else:
                    QMessageBox.critical(self, 'Error', str(self.analysis_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)            
            self.setDisabled(False)
            self.analysis_progress_bar.clear()

    def set_analysis_table_content(self):
        if self.analysis_type == 'Cluster algorithms':
            if self.analysis_status:
                self.analysis_tabWidget.setTabEnabled(0, True)
                self.analysis_tabWidget.setTabEnabled(1, True)
                self.analysis_tabWidget.setTabEnabled(2, False)
                self.analysis_tabWidget.setTabEnabled(3, False)

                self.analysis_status_label.setText('%s calculation complete.' % self.analysis_selected_algorithm)
                self.analysis_tableWidget.setColumnCount(2)
                self.analysis_tableWidget.setRowCount(self.analysis_data.row)
                self.analysis_tableWidget.setHorizontalHeaderLabels(['SampleName', 'Cluster'])
                for i in range(self.analysis_data.row):
                    cell = QTableWidgetItem(self.analysis_data.dataframe.index[i])
                    self.analysis_tableWidget.setItem(i, 0, cell)
                    cell1 = QTableWidgetItem(str(self.analysis_data.cluster_result[i]))
                    self.analysis_tableWidget.setItem(i, 1, cell1)
                """ plot with Matplotlib """
                self.analysis_diagram_layout.removeWidget(self.analysis_diagram)
                sip.delete(self.analysis_diagram)
                plot_data = self.analysis_data.generate_plot_data(self.analysis_data.cluster_result, self.analysis_data.cluster_plot_data)
                self.analysis_diagram = PlotWidgets.ClusteringDiagramMatplotlib()
                self.analysis_diagram.init_data('Clustering', plot_data)
                self.analysis_diagram_layout.addWidget(self.analysis_diagram)
                self.setDisabled(False)
            else:
                self.analysis_status_label.setText(str(self.analysis_data.error_msg))
                QMessageBox.critical(self, 'Calculate failed', str(self.analysis_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.setDisabled(False)
        elif self.analysis_type == 'Dimensionality reduction algorithms':
            if self.analysis_status:
                self.analysis_tabWidget.setTabEnabled(0, True)
                self.analysis_tabWidget.setTabEnabled(1, True)
                self.analysis_tabWidget.setTabEnabled(2, False)
                self.analysis_tabWidget.setTabEnabled(3, False)
                
                self.analysis_status_label.setText('%s calculation complete.' % self.analysis_selected_algorithm)                
                self.analysis_tableWidget.setColumnCount(self.analysis_default_para['n_components'] + 1)
                self.analysis_tableWidget.setRowCount(self.analysis_data.row)
                self.analysis_tableWidget.setHorizontalHeaderLabels(['SampleName'] + ['PC%s' % i for i in range(1, self.analysis_default_para['n_components'] + 1)])
                for i in range(self.analysis_data.row):
                    cell = QTableWidgetItem(self.analysis_data.dataframe.index[i])
                    self.analysis_tableWidget.setItem(i, 0, cell)
                    for j in range(self.analysis_default_para['n_components']):
                        cell = QTableWidgetItem(str(self.analysis_data.dimension_reduction_result[i][j]))
                        self.analysis_tableWidget.setItem(i, j+1, cell)                
                """ plot with Matploglib """
                self.analysis_diagram_layout.removeWidget(self.analysis_diagram)                
                sip.delete(self.analysis_diagram)
                plot_data = self.analysis_data.generate_plot_data(self.analysis_data.datalabel, self.analysis_data.cluster_plot_data)
                self.analysis_diagram = PlotWidgets.ClusteringDiagramMatplotlib()
                self.analysis_diagram.init_data('Dimension reduction', plot_data)
                self.analysis_diagram_layout.addWidget(self.analysis_diagram)
                self.setDisabled(False)
            else:
                self.analysis_status_label.setText(str(self.analysis_data.error_msg))
                QMessageBox.critical(self, 'Calculate failed', str(self.analysis_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.setDisabled(False)
        else:
            if self.analysis_status:
                self.analysis_tabWidget.setTabEnabled(0, False)
                self.analysis_tabWidget.setTabEnabled(1, False)
                self.analysis_tabWidget.setTabEnabled(2, True)
                self.analysis_tabWidget.setTabEnabled(3, True)
                    
                self.analysis_status_label.setText('%s calculation complete.' %self.analysis_selected_algorithm)
                
                tmp_init_data = np.hstack((self.analysis_data.dataframe.index.to_numpy().reshape((-1, 1)), self.analysis_data.feature_normalization_data.values[:, 1:]))                
                self.analysis_tableWidget_1.init_data(['Samples'] + list(self.analysis_data.feature_normalization_data.columns[1:]), tmp_init_data)
                """ plot with Matploglib """
                self.analysis_hist_layout.removeWidget(self.analysis_histogram)
                sip.delete(self.analysis_histogram)
                # data = self.analysis_data.feature_normalization_data.values[:, 1:]
                data = self.analysis_data.feature_normalization_data.values
                self.analysis_histogram = PlotWidgets.HistogramWidget2()
                self.analysis_histogram.init_data('Normalized data', data)
                self.analysis_hist_layout.addWidget(self.analysis_histogram)
                self.setDisabled(False)
            else:
                self.analysis_status_label.setText(str(self.analysis_data.error_msg))
                QMessageBox.critical(self, 'Calculate failed', str(self.analysis_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.setDisabled(False)

    def display_error_msg(self, err_msg):
        QMessageBox.critical(self, 'Error', str(err_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def display_warning_msg(self, err_msg):
        QMessageBox.warning(self, 'Warning', str(err_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    """ events in plot widgetUI """
    def plot_display_message(self, message):
        self.plot_status_label.setText(message)
    
    def plot_panel_clear(self):
        self.plot_data_file = None
        self.plot_data = None
        self.plot_file_lineEdit.setText('')
        self.plot_file_lineEdit2.setText('')   

    def plot_data_from_file(self):
        self.plot_data_file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        if ok:
            self.plot_data = DataAnalysis.ILearnData(self.analysis_default_para)
            ok1 = self.plot_data.load_data_from_file(self.plot_data_file)
            if ok1:
                self.plot_datashape.setText('Data shape: (%s, %s)' %(self.plot_data.row, self.plot_data.column))
                self.plot_file_lineEdit.setText(self.plot_data_file)
                self.plot_file_lineEdit2.setText('')  
            else:
                self.plot_panel_clear()
                QMessageBox.critical(self, 'Error', str(self.plot_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            self.plot_panel_clear()

    def plot_data_from_file2(self):
        self.plot_data_file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        if ok:
            self.plot_data = DataAnalysis.ILearnData(self.analysis_default_para)
            ok1 = self.plot_data.load_data_from_file2(self.plot_data_file)
            if ok1:
                self.plot_datashape.setText('Data shape: (%s, %s)' %(self.plot_data.row, self.plot_data.column))
                self.plot_file_lineEdit2.setText(self.plot_data_file)
                self.plot_file_lineEdit.setText('')
            else:
                self.plot_panel_clear()
                QMessageBox.critical(self, 'Error', str(self.plot_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            self.plot_panel_clear()
 
    def plot_tree_clicked(self, index):
        item = self.plot_treeWidget.currentItem()
        if item and item.text(0) not in ['Plot types']:
            self.selected_plot_type = item.text(0)            
            
    def draw_plot(self):
        if self.selected_plot_type is None or self.plot_data_file is None or self.plot_data is None:
            QMessageBox.critical(self, 'Error', 'Please check your input.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)        
        else:
            """ plot with Matploglib """
            if self.selected_plot_type == 'Histogram and kernel density plot':
                self.plot_view_layout.removeWidget(self.plot_plt)
                sip.delete(self.plot_plt)
                data = copy.deepcopy(self.plot_data.dataframe)
                if len(set(self.plot_data.datalabel)) > 1:
                    data.insert(0, 'Labels', self.plot_data.datalabel)
                    self.plot_plt = PlotWidgets.HistogramWidget2()
                if len(set(self.plot_data.datalabel)) == 1:
                    self.plot_plt = PlotWidgets.HistogramWidget()
                self.plot_plt.init_data('Data distribution', data.values)
                self.plot_view_layout.addWidget(self.plot_plt)
            if self.selected_plot_type == 'Box plot':
                self.plot_view_layout.removeWidget(self.plot_plt)
                sip.delete(self.plot_plt)
                data = copy.deepcopy(self.plot_data.dataframe)
                if len(set(self.plot_data.datalabel)) == 1:
                    self.plot_plt = PlotWidgets.BoxplotSpanSelector()
                elif len(set(self.plot_data.datalabel)) > 1:
                    data.insert(0, 'Labels', self.plot_data.datalabel)
                    self.plot_plt = PlotWidgets.BoxplotSpanSelector_multiSamples()
                else:
                    pass                
                self.plot_plt.init_data(data, 'Feature names', 'Value')
                self.plot_view_layout.addWidget(self.plot_plt)
            if self.selected_plot_type == 'Heatmap':
                self.plot_view_layout.removeWidget(self.plot_plt)
                sip.delete(self.plot_plt)
                data = copy.deepcopy(self.plot_data.dataframe)
                self.plot_plt = PlotWidgets.HeatMapSpanSelector()
                self.plot_plt.init_data(data, 'Features', 'Samples')
                self.plot_view_layout.addWidget(self.plot_plt)
            if self.selected_plot_type == 'Scatter plot':
                self.plot_view_layout.removeWidget(self.plot_plt)
                sip.delete(self.plot_plt)
                rd_data, _ = self.plot_data.PCA(2)               
                plot_data = self.plot_data.generate_plot_data(self.plot_data.datalabel, rd_data)
                self.plot_plt = PlotWidgets.ClusteringDiagramMatplotlib()
                self.plot_plt.init_data('', plot_data)
                self.plot_view_layout.addWidget(self.plot_plt)
            if self.selected_plot_type == 'Circular plot':
                self.plot_view_layout.removeWidget(self.plot_plt)
                sip.delete(self.plot_plt)
                data = copy.deepcopy(self.plot_data.dataframe)
                self.plot_plt = PlotWidgets.CircosWidget()
                self.plot_plt.init_data(data)
                self.plot_view_layout.addWidget(self.plot_plt)
                
def runiFeatureOmegaGUI():
    app = QApplication(sys.argv)
    window = IFeatureOmegaGui()
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IFeatureOmegaGui()
    # app.setFont(QFont('Arial', 10))
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())
