import deepchem as dc
from rdkit import Chem
import pandas as pd
import numpy as np
from utils import get_MACCS
from drug_util import drug_feature_extract
from sklearn.preprocessing import StandardScaler

def normlize(data):
    data_array = np.array(data, dtype='float32')
    # data_array.reshape(-1, 1)
    scaler = StandardScaler()
    norm = scaler.fit(data_array).transform(data_array)
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    return norm

def MinMaxScale(data):
    result = (data-data.min())/(data.max()-data.min())*100
    return result



def getData(dataset):
    if dataset == 'ONEIL':
        drug_chemical_file = '../../Data/ONEIL-COSMIC/drugfeature/drug_chemical.csv'
        drug_target_file = '../../Data/ONEIL-COSMIC/drugfeature/onehot_ON_2.csv'
        cline_feature_file = '../../Data/ONEIL-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = '../../Data/ONEIL-COSMIC/drug_synergy.csv'
        drug_smiles_file = '../../Data/ONEIL-COSMIC/drug_smiles.csv'
        drug = pd.read_csv(drug_chemical_file, sep=',', index_col=[0], header=0,
                                usecols=['cid', 'mw', 'polararea', 'complexity', 'xlogp', 'heavycnt', 'hbonddonor',
                                         'hbondacc',
                                         'rotbonds', 'exactmass', 'monoisotopicmass', 'charge', 'covalentunitcnt',
                                         'isotopeatomcnt', 'totalatomstereocnt', 'definedatomstereocnt',
                                         'undefinedatomstereocnt',
                                         'totalbondstereocnt', 'definedbondstereocnt', 'undefinedbondstereocnt'])
        drug_data_chem = normlize(drug)
        drug_target = pd.read_csv(drug_target_file, sep=',', index_col=[0], header=0)
        drug_data_target = np.array(drug_target, dtype='float32')
    elif dataset == 'ALMANAC':
        drug_chemical_file = '../../Data/ALMANAC-COSMIC/drugfeature/drug_chemical.csv'
        drug_target_file = '../../Data/ALMANAC-COSMIC/drugfeature/onehot2.csv'
        cline_feature_file = '../../Data/ALMANAC-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = '../../Data/ALMANAC-COSMIC/drug_synergy.csv'
        drug_smiles_file = '../../Data/ALMANAC-COSMIC/drug_smiles.csv'
        drug = pd.read_csv(drug_chemical_file, sep=',', index_col=[0], header=0,
                           usecols=['cid', 'mw', 'polararea', 'complexity', 'xlogp', 'heavycnt', 'hbonddonor',
                                    'hbondacc',
                                    'rotbonds', 'exactmass', 'monoisotopicmass', 'charge', 'covalentunitcnt',
                                    'isotopeatomcnt', 'totalatomstereocnt', 'definedatomstereocnt',
                                    'undefinedatomstereocnt',
                                    'totalbondstereocnt', 'definedbondstereocnt', 'undefinedbondstereocnt'])
        drug_data_chem = normlize(drug)
        drug_target = pd.read_csv(drug_target_file, sep=',', index_col=[0], header=0)
        drug_data_target = np.array(drug_target, dtype='float32')


        # drug_data = np.hstack((drug_data_chem,drug_data_target))

    else:
        drug_target_file = '../../Data/ALMANAC-COSMIC/drugfeature/onehot2.csv'
        cline_feature_file = '../../Data/ALMANAC-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = '../../Data/ALMANAC-COSMIC/drug_synergy.csv'
        drug = pd.read_csv(drug_target_file, sep=',', index_col=[0], header=0)
        drug_data = np.array(drug,dtype='float32')
        drug_smiles_file = '../../Data/ALMANAC-COSMIC/drug_smiles.csv'
    drug_smile = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])
    drug_data = pd.DataFrame()
    drug_smiles_fea = []
    featurizer = dc.feat.ConvMolFeaturizer()
    for tup in zip(drug_smile['pubchemid'], drug_smile['isosmiles']):
        mol = Chem.MolFromSmiles(tup[1])
        mol_f = featurizer.featurize(mol)
        drug_data[str(tup[0])] = [mol_f[0].get_atom_features(), mol_f[0].get_adjacency_list()]
        drug_smiles_fea.append(get_MACCS(tup[1]))
    drug_num = len(drug.index)
    d_map = dict(zip(drug.index, range(drug_num)))
    drug_fea = drug_feature_extract(drug_data)
    gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    cline_num = len(gene_data.index)
    # c_map = dict(zip(gene_data.index, range(drug_num, drug_num + cline_num)))
    c_map = dict(zip(gene_data.index, range(drug_num, drug_num + cline_num)))
    cline_fea = np.array(gene_data, dtype='float32')
    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)
    synergy = [[d_map[row[0]], d_map[row[1]], c_map[row[2]], float(row[3]), float(row[3])] for index, row in
               synergy_load.iterrows() if (row[0] in drug.index and row[1] in drug.index and
                                           str(row[2]) in gene_data.index)]
    # return cline_fea, drug_smiles_fea, drug_data_chem,drug_data_target,gene_data, synergy
    return cline_fea, drug_smiles_fea, drug_fea,drug_data_chem,drug_data_target,gene_data, synergy


