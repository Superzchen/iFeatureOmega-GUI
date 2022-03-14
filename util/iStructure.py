#!/usr/bin/env python
# _*_ coding: utf-8 _*_


import sys, re, os
import numpy as np
import pandas as pd
import networkx as nx
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.HSExposure import HSExposureCA, HSExposureCB
from Bio.PDB.PDBList import PDBList

AA_3to1 = {
    'GLY': 'G',
    'ALA': 'A',
    'LEU': 'L',
    'ILE': 'I',
    'VAL': 'V',
    'PRO': 'P',
    'PHE': 'F',
    'MET': 'M',
    'TRP': 'W',
    'SER': 'S',
    'GLN': 'Q',
    'THR': 'T',
    'CYS': 'C',
    'ASN': 'N',
    'TYR': 'Y',
    'ASP': 'D',
    'GLU': 'E',
    'LYS': 'K',
    'ARG': 'R',
    'HIS': 'H',
}

AA_1to3 = dict(zip(AA_3to1.values(), AA_3to1.keys()))

AA_group = {
    'G': 'aliphatic',
    'A': 'aliphatic',
    'V': 'aliphatic',
    'L': 'aliphatic',
    'M': 'aliphatic',
    'I': 'aliphatic',
    'F': 'aromatic',
    'Y': 'aromatic',
    'W': 'aromatic',
    'K': 'positive charged',
    'R': 'positive charged',
    'H': 'positive charged',
    'D': 'negative charged',
    'E': 'negative charged',
    'S': 'uncharged',
    'T': 'uncharged',
    'C': 'uncharged',
    'P': 'uncharged',
    'N': 'uncharged',
    'Q': 'uncharged',
}

AA_HEC = {
    'H': 'H',
    'B': 'E',
    'E': 'E',
    'G': 'H',
    'I': 'H',
    'T': 'C',
    'S': 'C',
    '-': 'C'
}

class Structure(object):
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.protein = None
        self.structure = None
        self.model = None
        self.encodings = None
        self.error_msg = None
        self.pdb_id = None

    def read_pdb(self):
        if self.pdb_file.endswith('.pdb') or self.pdb_file.endswith('.PDB'):
            self.protein = PDBParser(PERMISSIVE=1)
            self.structure = self.protein.get_structure(self.pdb_file[0:4], self.pdb_file)            
        elif self.pdb_file.endswith('.cif') or self.pdb_file.endswith('.CIF'):
            parser = MMCIFParser()
            self.structure = parser.get_structure(self.pdb_file[0:4], self.pdb_file)            
        try:
            self.model = self.structure[0]           
            return True
        except KeyError as e:
            self.error_msg = str(e)
            return False

    def get_pdb_id(self):
        try:
            if self.pdb_file.lower().endswith('.pdb'):
                with open(self.pdb_file) as f:
                    record = f.read().strip().split('\n')[0]
                tmp = re.split('\s+', record)               
                self.pdb_id = tmp[3].lower()
            if self.pdb_file.lower().endswith('.cif'):
                with open(self.pdb_file) as f:
                    self.pdb_id = f.read().strip().split('\n')[0].split('_')[1].lower()
            return True, self.pdb_id
        except Exception as e:
            self.error_msg = str(e)
            return False, None

    def get_residue_descriptor(self, target_list, method='AAC_type1', shell=(3, 20, 2)):   # target_list: 2d list, [[chain, resseq, resname], [chain, resseq, resname]]
        try:
            tmp_residues = list(self.model.get_residues())
            dssp_status = True
            try:
                dssp = DSSP(self.model, self.pdb_file)  # calculate secondary structure
            except Exception as e:
                self.error_msg = 'Secondary structure calculate failed. Please check whether DSSP was installed?'
                dssp_status = 0
                dssp = []
            
            residues = []  # remove hetfield, only residues are saved.
            for residue in tmp_residues:
                if residue.has_id('CA') or residue.has_id('CB'):
                    residue_id = residue.get_id()
                    if residue_id[0] == ' ':
                        residues.append(residue)
            
            if len(target_list) == 0:  # if target_list is empty, take all the residue as target
                for residue in residues:
                    target_list.append([residue.parent.id, residue.id[1], residue.resname])
            
            print(len(residues), len(dssp))

            header = []
            encodings = []
            if len(residues) == len(dssp):
                for item in target_list:
                    target_chain, target_resseq, target_resname = item[0], item[1], item[2]
                    df_residue = None
                    if self.model.has_id(target_chain) and self.model[target_chain].has_id(target_resseq) and self.model[target_chain][target_resseq].get_resname() == target_resname:  # check if the item in target_list exist
                        target_residue = self.model[target_chain][target_resseq]
                        target_residue_name = target_residue.get_resname()
                        target_atom = target_residue['CB'] if target_residue.has_id('CB') else target_residue['CA']

                        record_list = []  # [chain, resseq, resname, distance, property, hec8, hec3]
                        for residue, hec in zip(residues, dssp):
                            source_atom = residue['CB'] if residue.has_id('CB') else residue['CA']
                            record_list.append([residue.parent.get_id(), residue.get_id()[1], AA_3to1[residue.get_resname()], target_atom - source_atom, AA_group[AA_3to1[residue.get_resname()]], hec[2], AA_HEC[hec[2]]])
                            # print([residue.parent.get_id(), residue.get_id()[1], AA_3to1[residue.get_resname()], target_atom - source_atom, AA_group[AA_3to1[residue.get_resname()]], hec[2], AA_HEC[hec[2]]])

                        df_residue = pd.DataFrame(np.array(record_list), columns=['chain', 'resseq', 'resname', 'distance', 'property', 'hec8', 'hec3'])
                        df_residue['distance'] = df_residue['distance'].astype('float64')                    
                    
                        if method == 'AAC_type1':
                            _, tmp_code = self.AAC_type1(df_residue, shell)
                        elif method == 'AAC_type2':
                            _, tmp_code = self.AAC_type2(df_residue, shell)
                        elif method == 'GAAC_type1':
                            _, tmp_code = self.GAAC_type1(df_residue, shell)
                        elif method == 'GAAC_type2':
                            _, tmp_code = self.GAAC_type2(df_residue, shell)
                        elif method == 'SS8_type1':
                            _, tmp_code = self.SS8_type1(df_residue, shell)
                        elif method == 'SS8_type2':
                            _, tmp_code = self.SS8_type2(df_residue, shell)
                        elif method == 'SS3_type1':
                            _, tmp_code = self.SS3_type1(df_residue, shell)
                        elif method == 'SS3_type2':
                            _, tmp_code = self.SS3_type2(df_residue, shell)
                        else:
                            return False, None
                        encodings.append([target_chain + '_' + target_resname + '_' + str(target_resseq)] + tmp_code)
                        header = ['Sample'] + _
            else:
                self.error_msg = 'Secondary structure calculate failed.'
                for item in target_list:
                    target_chain, target_resseq, target_resname = item[0], item[1], item[2]
                    if self.model.has_id(target_chain) and self.model[target_chain].has_id(target_resseq) and self.model[target_chain][target_resseq].get_resname() == target_resname:  # check if the item in target_list exist
                        target_residue = self.model[target_chain][target_resseq]
                        target_residue_name = target_residue.get_resname()
                        target_atom = target_residue['CB'] if target_residue.has_id('CB') else target_residue['CA']

                        record_list = []  # [chain, resseq, resname, distance, property]
                        for residue in residues:
                            source_atom = residue['CB'] if residue.has_id('CB') else residue['CA']
                            record_list.append([residue.parent.get_id(), residue.get_id()[1], AA_3to1[residue.get_resname()], target_atom - source_atom, AA_group[AA_3to1[residue.get_resname()]]])                        
                            
                        df_residue = pd.DataFrame(np.array(record_list), columns=['chain', 'resseq', 'resname', 'distance', 'property'])
                        df_residue['distance'] = df_residue['distance'].astype('float64')

                        if method == 'AAC_type1':
                            _, tmp_code = self.AAC_type1(df_residue, shell)                    
                        elif method == 'AAC_type2':
                            _, tmp_code = self.AAC_type2(df_residue, shell)
                        elif method == 'GAAC_type1':
                            _, tmp_code = self.GAAC_type1(df_residue, shell)
                        elif method == 'GAAC_type2':
                            _, tmp_code = self.GAAC_type2(df_residue, shell)                    
                        else:
                            return False, None
                        encodings.append([target_chain + '_' + target_resname + '_' + str(target_resseq)] + tmp_code)                        
                        header = ['Sample'] + _
            
            

            df_encodings = pd.DataFrame(np.array(encodings), columns=header)
            return True, df_encodings
        except Exception as e:
            print(e)
            self.error_msg = str(e)
            return False, None

    def get_atom_descriptor(self, target_list, method='AC_type1', shell=(1, 10, 1)):       # target_list: 2d list, [[chain, serial_number, name], [chain, serial_number, name]]
        try:
            tmp_atoms = self.model.get_atoms()
            atoms = {}                              # use dict
            for atom in tmp_atoms:
                if atom.parent.get_id()[0] != 'W':  # reomove water residue
                    atoms[atom.parent.parent.id + str(atom.serial_number) + re.sub(' ', '', atom.element)] = atom
            
            if len(target_list) == 0:
                for atom in atoms:
                    if atoms[atom].name == 'CA':
                        target_list.append([atoms[atom].parent.parent.id, atoms[atom].serial_number, re.sub(' ', '', atoms[atom].element)])
            
            header = []
            encodings = []
            for item in target_list:
                target_chain, target_serial_number, target_element = item[0], item[1], item[2]
                if target_chain + str(target_serial_number) + target_element in atoms:
                    target = atoms[target_chain + str(target_serial_number) + target_element]                
                    record_list = []  # [chain, serial_number, element, distance]
                    for key in atoms:
                        atom = atoms[key]
                        record_list.append([atom.parent.parent.id, atom.serial_number, re.sub(' ', '', atom.element), target - atom])
                        # print(atom.parent.parent.id, atom.serial_number, re.sub(' ', '', atom.element), target - atom)                  

                    df_atom = pd.DataFrame(np.array(record_list), columns=['chain', 'serial_number', 'element', 'distance'])
                    df_atom['distance'] = df_atom['distance'].astype('float64')

                    if method == 'AC_type1':
                        _, tmp_code = self.AC_type1(df_atom, shell)                    
                    elif method == 'AC_type2':
                        _, tmp_code = self.AC_type2(df_atom, shell)
                    else:
                        return False, None
                    encodings.append([target_chain + '_' + re.sub(' ', '', target_element) + '_' + str(target_serial_number)] + tmp_code)                        
                    header = ['Sample'] + _                

            df_encodings = pd.DataFrame(np.array(encodings), columns=header)
            return True, df_encodings
        except Exception as e:
            self.error_msg = str(e)
            return False, None

    def get_residue_depth(self):
        msms_status = True
        try:
            rd = ResidueDepth(self.model)
        except Exception as e:
            self.error_msg = 'Residue depth calculate failed. Please check whether msms was installed?'
            msms_status = 0
            return False, None

        key_array = rd.keys()
        
        encodings = []
        for key in key_array:
            residue_depth, ca_depth = rd[key]            
            encodings.append([key[0]+'_'+str(key[1][1]), residue_depth, ca_depth])

        df_encodings = pd.DataFrame(np.array(encodings), columns=['Sample', 'Residue_depth', 'CA_depth'])
        print(df_encodings)
        return True, df_encodings
       
    def get_HSE_CA(self):
        try:
            hse = HSExposureCA(self.model)
            encodings = []
            for e in hse:                
                encodings.append([e[0].parent.id+'_'+e[0].resname+'_'+str(e[0].id[1]), e[1][0], e[1][1], e[1][2]])
            df_encodings = pd.DataFrame(np.array(encodings), columns=['Sample', 'HSE_CA_value1', 'HSE_CA_value2', 'HSE_CA_value3'])
            return True, df_encodings
        except Exception as e:
            self.error_msg = str(e)
            return False, None
    
    def get_HSE_CB(self):
        try:
            hse = HSExposureCB(self.model)
            encodings = []
            for e in hse:
                encodings.append([e[0].parent.id+'_'+e[0].resname+'_'+str(e[0].id[1]), e[1][0], e[1][1], e[1][2]])
            df_encodings = pd.DataFrame(np.array(encodings), columns=['Sample', 'HSE_CB_value1', 'HSE_CB_value2', 'HSE_CB_value3'])
            return True, df_encodings
        except Exception as e:
            self.error_msg = str(e)
            return False, None

    def AAC_type1(self, df, shell=(3, 20, 2)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]            
            
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            AA_dict = {}
            for aa in AA:
                header.append('shell_%s.%s' %(m, aa))
                AA_dict[aa] = 0                
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i,2] in AA_dict:
                    AA_dict[df_tmp.iloc[i,2]] += 1
            for key in AA_dict:
                if len(df_tmp) == 0:
                    AA_dict[key] = 0
                else:
                    AA_dict[key] /= len(df_tmp)
        
            code += AA_dict.values()
            m += 1
        return header, code
    
    def AAC_type2(self, df, shell=(3, 20, 2)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]            
            
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            AA_dict = {}
            for aa in AA:
                header.append('shell_%s.%s' %(m, aa))
                AA_dict[aa] = 0                
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i,2] in AA_dict:
                    AA_dict[df_tmp.iloc[i,2]] += 1
            for key in AA_dict:
                if len(df_tmp) == 0:
                    AA_dict[key] = 0
                else:
                    AA_dict[key] /= len(df_tmp)
        
            code += AA_dict.values()
            m += 1
        return header, code

    def GAAC_type1(self, df, shell=(3, 20, 2)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]
            group_dict = {
                'aliphatic': 0,
                'aromatic': 0,
                'positive charged': 0,
                'negative charged': 0,
                'uncharged': 0
            }
            group_list = ['aliphatic', 'aromatic', 'positive charged', 'negative charged', 'uncharged']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 4] in group_dict:
                    group_dict[df_tmp.iloc[i, 4]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code

    def GAAC_type2(self, df, shell=(3, 20, 2)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]
            group_dict = {
                'aliphatic': 0,
                'aromatic': 0,
                'positive charged': 0,
                'negative charged': 0,
                'uncharged': 0
            }
            group_list = ['aliphatic', 'aromatic', 'positive charged', 'negative charged', 'uncharged']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 4] in group_dict:
                    group_dict[df_tmp.iloc[i, 4]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code

    def SS8_type1(self, df, shell=(3, 20, 2)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]
            group_dict = {
                'H': 0,
                'B': 0,
                'E': 0,
                'G': 0,
                'I': 0,
                'T': 0,
                'S': 0,
                '-': 0
            }
            group_list = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 5] in group_dict:
                    group_dict[df_tmp.iloc[i, 5]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code
    
    def SS8_type2(self, df, shell=(3, 20, 2)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]
            group_dict = {
                'H': 0,
                'B': 0,
                'E': 0,
                'G': 0,
                'I': 0,
                'T': 0,
                'S': 0,
                '-': 0
            }
            group_list = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 5] in group_dict:
                    group_dict[df_tmp.iloc[i, 5]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code
    
    def SS3_type1(self, df, shell=(3, 20, 2)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]
            group_dict = {
                'H': 0,
                'B': 0,
                'E': 0,
                'G': 0,
                'I': 0,
                'T': 0,
                'S': 0,
                '-': 0
            }
            group_list = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 6] in group_dict:
                    group_dict[df_tmp.iloc[i, 6]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code
    
    def SS3_type2(self, df, shell=(3, 20, 2)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]
            group_dict = {
                'H': 0,
                'B': 0,
                'E': 0,
                'G': 0,
                'I': 0,
                'T': 0,
                'S': 0,
                '-': 0
            }
            group_list = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
            for g in group_list:
                header.append('shell_%s.%s' %(m, g))
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 6] in group_dict:
                    group_dict[df_tmp.iloc[i, 6]] += 1
            for key in group_dict:
                if len(df_tmp) == 0:
                    group_dict[key] = 0
                else:
                    group_dict[key] /= len(df_tmp)
            for key in group_list:
                code.append(group_dict[key]) 
            m+=1
        return header, code

    def AC_type1(self, df, shell=(1, 10, 1)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[(df['distance'] >= s) & (df['distance'] < s+shell[2])]
            AA = 'CNOS'
            AA_dict = {}
            for aa in AA:
                header.append('shell_%s.%s' %(m, aa))
                AA_dict[aa] = 0
            
            sum = 0
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 2] in AA_dict:
                    AA_dict[df_tmp.iloc[i, 2]] += 1
                    sum += 1
            for key in AA_dict:
                if sum == 0:
                    AA_dict[key] = 0
                else:
                    AA_dict[key] /= sum
            
            code += AA_dict.values()
            m += 1           
        return header, code
    
    def AC_type2(self, df, shell=(1, 10, 1)):
        header = []
        code = []
        m = 1
        for s in range(shell[0], shell[1]+1, shell[2]):
            df_tmp = df[df['distance'] < s+shell[2]]
            AA = 'CNOS'
            AA_dict = {}
            for aa in AA:
                header.append('shell_%s.%s' %(m, aa))
                AA_dict[aa] = 0
            
            sum = 0
            for i in range(len(df_tmp)):
                if df_tmp.iloc[i, 2] in AA_dict:
                    AA_dict[df_tmp.iloc[i, 2]] += 1
                    sum += 1
            for key in AA_dict:
                if sum == 0:
                    AA_dict[key] = 0
                else:
                    AA_dict[key] /= sum
            
            code += AA_dict.values()
            m += 1           
        return header, code

    def get_network_descriptor(self, target_list, distance_cutoff=11):
        try:
            tmp_residues = list(self.model.get_residues())        
            residues = []
            for residue in tmp_residues:   # remove hetfield, only residues are saved.
                residue_id = residue.get_id()
                if residue_id[0] == ' ':
                    residues.append(residue)
            
            if len(target_list) == 0:  # if target_list is empty, take all the residue as target
                for residue in residues:
                    target_list.append([residue.parent.id, residue.id[1], residue.resname])

            G = nx.Graph()     
            # add graph nodes
            for i in range(len(residues)):
                node = residues[i].resname + '_' + residues[i].parent.id + str(residues[i].id[1])
                G.add_node(node)

            CB_coord_dict = {}
            
            # add graph edges
            for i in range(len(residues)):
                node = residues[i].resname + '_' + residues[i].parent.id + str(residues[i].id[1])            
                atom = residues[i]['CB'] if residues[i].has_id('CB') else residues[i]['CA']
                CB_coord_dict[node] = atom.coord            
                for j in range(i+1, len(residues)):
                    node_2 = residues[j].resname + '_' + residues[j].parent.id + str(residues[j].id[1])
                    atom_2 = residues[j]['CB'] if residues[j].has_id('CB') else residues[j]['CA']
                    distance = atom - atom_2
                    if distance <= distance_cutoff:
                        G.add_edge(node, node_2)
            
            net_dict = {}
            net_dict['average_clustering'] = nx.average_clustering(G)        
            net_dict['diameter'] = nx.diameter(G)
            net_dict['average_shortest_path_length'] = nx.average_shortest_path_length(G)
                
            net_dict['degree_centrality'] = nx.degree_centrality(G)                       # degree centrality        
            net_dict['betweenness_centrality'] = nx.betweenness_centrality(G)             # betweenness
            net_dict['clustering'] = nx.clustering(G)                                     # clustering coefficient
            net_dict['closeness_centrality'] = nx.closeness_centrality(G)                 # closeness
            net_dict['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(G)       # centrality        

            encodings = []
            for item in target_list:
                id = item[2]+'_'+item[0]+str(item[1])
                tmp_code = [id]
                if id in G.nodes:
                    tmp_code += [G.degree(id), net_dict['degree_centrality'].get(id, 'NA'), net_dict['betweenness_centrality'].get(id, 'NA'), net_dict['clustering'].get(id, 'NA'), net_dict['closeness_centrality'].get(id, 'NA'), net_dict['eigenvector_centrality'].get(id, 'NA')]            
                    encodings.append(tmp_code)       

            df_encodings = pd.DataFrame(np.array(encodings), columns=['Sample', 'degree', 'degree_centrality', 'betweenness', 'clustering_coefficient', 'closeness', 'centrality'])        
            
            # generate network gexf file
            # chains = self.generate_gexf(G, CB_coord_dict, [item[2]+'_'+item[0]+str(item[1]) for item in target_list])          

            return True, df_encodings
        except Exception as e:
            self.error_msg = str(e)
            return False, None, None
        
    def generate_network(self, Graph, target_list):
        # 此段代码需要进一步修改，生成的json文件，最后一条记录后面不能带有逗号
        node_dict = {}
        for i, node in enumerate(Graph.nodes):
            node_dict[node] = i
        with open('network.json', 'w') as f:
            f.write('{\n')
            f.write('    "type": "force",\n')
            # write categories
            f.write('    "categories": [\n')
            f.write('        {\n            "name": "Residues",\n            "keyword": {},\n            "base": "Residues"\n        },\n')
            f.write('        {\n            "name": "TargetResidues",\n            "keyword": {},\n            "base": "TargetResidues"\n        }\n')
            f.write('    ],\n')
            
            # write nodes
            f.write('    "nodes": [\n')
            for node in Graph.nodes:
                f.write('        {\n            "name": "%s",\n            "value": 1,\n            "category": %s\n        },\n' %(node, 1 if node in target_list else 0))
            f.write('    ],\n')

            # write edges
            f.write('    "links": [\n')
            for edge in Graph.edges:
                f.write('        {\n            "source": %s,\n            "target": %s\n        },\n' %(node_dict[edge[0]], node_dict[edge[1]]))
            f.write('    ],\n')
            f.write('}')
    
    def generate_gexf(self, Graph, CB_coord_dict, target_list):
        chains = self.model.get_list()
        chain_dict = {}
        m = 1
        chain_list = []
        for chain in chains:
            chain_list.append(chain.id)
            chain_dict[chain.id] = m
            m += 1

        node_dict = {}
        for i, node in enumerate(Graph.nodes):
            node_dict[node] = i
        with open('network.gexf', 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2" xmlns:viz="http://www.gexf.net/1.2draft/viz" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd">\n')
            f.write('  <meta lastmodifieddate="2021-01-08">\n    <creator>Gephi 0.8.1</creator>\n    <description></description>\n  </meta>\n')
            f.write('  <graph defaultedgetype="undirected" mode="static">\n')
            f.write('    <attributes class="node" mode="static">\n      <attribute id="modularity_class" title="Modularity Class" type="integer"/>\n    </attributes>\n')
            # write node information
            f.write('    <nodes>\n')
            for node in Graph.nodes:
                node_value = 0
                if node in target_list:
                    node_value = 0
                else:
                    node_value = chain_dict[node[4]]     
                f.write('      <node id="%s" label="%s">\n        <attvalues>\n          <attvalue for="modularity_class" value="%s"></attvalue>\n      </attvalues>\n        <viz:position x="%s" y="%s" z="%s"></viz:position>\n        <viz:color r="235" g="81" b="72"></viz:color>\n      </node>\n' %(node_dict[node], node, node_value, CB_coord_dict[node][0], CB_coord_dict[node][1], CB_coord_dict[node][2]))
            f.write('    </nodes>\n')
            # write edge information
            f.write('    <edges>\n')
            for i, edge in enumerate(Graph.edges):
                f.write('      <edge id="%s" source="%s" target="%s" weight="3.0">\n        <attvalues></attvalues>\n      </edge>\n' %(i, node_dict[edge[0]], node_dict[edge[1]]))
            f.write('    </edges>\n')            
            f.write('  </graph>\n')
            f.write('</gexf>\n')    
        
        return chain_list

    def save_descriptor(self, data, file_name):
        try:
            if not data is None:
                if file_name.endswith(".tsv"):
                    np.savetxt(file_name, data.values[:, 1:], fmt='%s', delimiter='\t')                    
                    return True
                if file_name.endswith(".tsv_1"):                    
                    data.to_csv(file_name, sep='\t', header=True, index=False)
                    return True
                if file_name.endswith(".csv"):
                    np.savetxt(file_name, data.values[:, 1:], fmt='%s', delimiter=',')
                    return True
                if file_name.endswith(".svm"):
                    with open(file_name, 'w') as f:
                        for line in data.values:
                            f.write('0')
                            for i in range(1, len(line)):
                                f.write('  %d:%s' % (i, line[i]))
                            f.write('\n')
                    return True
                if file_name.endswith(".arff"):
                    with open(file_name, 'w') as f:
                            f.write('@relation descriptor\n\n')
                            for i in data.columns:
                                f.write('@attribute %s numeric\n' % i)
                            f.write('@attribute play {yes, no}\n\n')
                            f.write('@data\n')
                            for line in data.values:
                                for fea in line[1:]:
                                    f.write('%s,' % fea)
                                f.write('no\n')
                    return True
            else:
                return False
        except Exception as e:
            return False

           
         
if __name__ == '__main__':
    pdb = Structure('../data/1iir.pdb')
    if pdb.read_pdb():
        target_list = [
            ['A', 1, 'MET'],
            ['A', 3, 'VAL'],
            ['A', 2, 'MET']
        ]
        # target_list = []

        # target_list =[
        #     ['A', 1, 'SER'],
        #     ['A', 2, 'ASN']
        # ]
        # pdb.get_network_descriptor(target_list)
        # df = pdb.get_residue_depth()
        # print(df)
        

        ok, _ = pdb.get_residue_descriptor(target_list, 'SS3_type2')
        print(ok, pdb.error_msg)
        print(_)

        # target_list = [
        #     ['A', 1, 'N'],
        #     ['A', 5, 'C']
        # ]
        # target_list = []
        # ok, res = pdb.get_atom_descriptor(target_list, method='AC_type2')
        # print(ok)
        # print(res)

    else:
        print('Error.')
    