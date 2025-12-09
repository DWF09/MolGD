from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
import copy
import numpy as np
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit import Chem, DataStructs
from .sascorer import compute_sa_score
from rdkit import Chem
import numpy as np
import torch
import pickle
import os
from .bond_analyze import get_bond_order, geom_predictor, allowed_bonds, allowed_fc_bonds


# 化学属性评估
def get_drug_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = Crippen.MolLogP(mol)
    lipinski = obey_lipinski(mol)
    return {
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski
    }

def obey_lipinski(mol):
    # mol = deepcopy(mol)
    # Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_info = dataset_info
        
    def evaluate(self, mols, mol_stable_list, train_smiles):
        
        valid_smiles = []
        chem_metrics = {
            'qed': [],
            'sa': [],
            'logp': [],
            'lipinski': [],
            'smiles': []
        }
        complete_n = 0
        
        for i, mol in enumerate(mols):
            
            mol = build_molecule(*mol, self.dataset_info)
            smiles = mol2smiles(mol)
            
            # qed, sa
            if mol_stable_list[i]==0 or smiles is None:
                metrics = dict(qed=0, sa=0, logp=0, lipinski=0, smiles=None)
            else:
                try:
                    metrics = get_drug_chem(mol)
                    metrics["smiles"] = smiles
                except:
                    metrics = dict(qed=0, sa=0, logp=0, lipinski=0, smiles=None)
            for key in chem_metrics:
                chem_metrics[key].append(metrics[key])
            
            # validity, complete, unique
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                except:
                    continue
                if len(mol_frags) == 1:
                    complete_n += 1
                    
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid_smiles.append(smiles)
        

        Validity = len(valid_smiles) / len(mols)
        Complete = complete_n / len(mols)
        
        if Validity > 0:
            # unique & valid rate
            Unique = len(set(valid_smiles)) / len(mols)
        else:
            Unique = 0

        Novelty = -1
        if train_smiles is not None:

            gen_smiles_set = set(valid_smiles) - {None}
            train_set = set(train_smiles) - {None}
            Novelty = len(gen_smiles_set - train_set) / len(mols)

        return dict(
            Validity=Validity,
            Complete=Complete,
            Unique=Unique,
            Novelty=Novelty,
            ChemMetrics=chem_metrics
        )
  
    
def build_xae_molecule(positions, atom_types, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if 'QM9' in dataset_info['name']:
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif 'Geom' in dataset_info['name']:
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E

def build_molecule(positions, atom_types, dataset_info):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    return mol













# class BasicMolecularMetrics(object):
#     def __init__(self, dataset_info, dataset_smiles_list=None):
#         self.atom_decoder = dataset_info['atom_decoder']
#         self.dataset_smiles_list = dataset_smiles_list
#         self.dataset_info = dataset_info

#         # Retrieve dataset smiles only for qm9 currently.
#         if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
#             self.dataset_smiles_list = retrieve_qm9_smiles(
#                 self.dataset_info)

#     def compute_validity(self, generated):
#         """ generated: list of couples (positions, atom_types)"""
#         valid = []

#         for graph in generated:
#             mol = build_molecule(*graph, self.dataset_info)
#             smiles = mol2smiles(mol)
#             if smiles is not None:
#                 mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
#                 largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
#                 smiles = mol2smiles(largest_mol)
#                 valid.append(smiles)

#         return valid, len(valid) / len(generated)

#     def compute_uniqueness(self, valid):
#         """ valid: list of SMILES strings."""
#         return list(set(valid)), len(set(valid)) / len(valid)

#     def compute_novelty(self, unique):
#         num_novel = 0
#         novel = []
#         for smiles in unique:
#             if smiles not in self.dataset_smiles_list:
#                 novel.append(smiles)
#                 num_novel += 1
#         return novel, num_novel / len(unique)

#     def evaluate(self, generated):
#         """ generated: list of pairs (positions: n x 3, atom_types: n [int])
#             the positions and atom types should already be masked. """
#         valid, validity = self.compute_validity(generated)
#         print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
#         if validity > 0:
#             unique, uniqueness = self.compute_uniqueness(valid)
#             print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

#             if self.dataset_smiles_list is not None:
#                 _, novelty = self.compute_novelty(unique)
#                 print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
#             else:
#                 novelty = 0.0
#         else:
#             novelty = 0.0
#             uniqueness = 0.0
#             unique = None
#         return [validity, uniqueness, novelty], unique


# def compute_validity(rdmols):
#     valid = []

#     for mol in rdmols:
#         smiles = mol2smiles(mol)
#         if smiles is not None:
#             valid.append(smiles)

#     return valid, len(valid) / len(rdmols)


def eval_rdmol(rd_mols, mol_stable_list, train_smiles=None):
    # validity and complete rate
    valid_smiles = []
    chem_metrics = {
        'qed': [],
        'sa': [],
        'logp': [],
        'lipinski': [],
        'smiles': []
    }

    complete_n = 0
    for i, mol in enumerate(rd_mols):
        mol = copy.deepcopy(mol)
        smiles = mol2smiles(mol)
        
        if mol_stable_list[i]==0 or smiles is None:
            metrics = dict(qed=0, sa=0, logp=0, lipinski=0, smiles=None)
        else:
            try:
                metrics = get_drug_chem(mol)
                metrics["smiles"] = smiles
            except:
                metrics = dict(qed=0, sa=0, logp=0, lipinski=0, smiles=None)
        for key in chem_metrics:
            chem_metrics[key].append(metrics[key])
        
        if smiles is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            except:
                continue
            if len(mol_frags) == 1:
                complete_n += 1
                
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)
            valid_smiles.append(smiles)
    

    Validity = len(valid_smiles) / len(rd_mols)
    Complete = complete_n / len(rd_mols)
    
    if Validity > 0:
        # unique & valid rate
        Unique = len(set(valid_smiles)) / len(rd_mols)
    else:
        Unique = 0

    Novelty = -1
    if train_smiles is not None:
        # num_novel = 0
        # for smiles in set(valid_smiles):
        #     if smiles not in train_smiles:
        #         num_novel += 1
        # Novelty = num_novel / len(rd_mols)
        gen_smiles_set = set(valid_smiles) - {None}
        train_set = set(train_smiles) - {None}
        Novelty = len(gen_smiles_set - train_set) / len(rd_mols)

    return dict(
        Validity=Validity,
        Complete=Complete,
        Unique=Unique,
        Novelty=Novelty,
        ChemMetrics=chem_metrics
    )


def get_rdkit_rmsd(mols, n_conf=32, random_seed=42, num_workers=16):
    # check the best alignment between generated mols and rdkit conformers

    lowest_rmsd = []
    for mol in mols:
        mol_3d = copy.deepcopy(mol)
        try:
            Chem.SanitizeMol(mol_3d)
        except:
            continue
        confIds = AllChem.EmbedMultipleConfs(mol_3d, n_conf, randomSeed=random_seed,
                                             clearConfs=True, numThreads=num_workers)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol_3d, numThreads=num_workers)
        except:
            continue
        tmp_rmsds = []
        for confId in confIds:
            # AllChem.UFFOptimizeMolecule(mol, confId=confId)
            # try:
            #     AllChem.MMFFOptimizeMolecule(mol_3d, confId=confId)
            # except:
            #     continue
            try:
                rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol_3d, refId=confId)
                tmp_rmsds.append(rmsd)
            except:
                continue

        if len(tmp_rmsds) != 0:
            lowest_rmsd.append(np.min(np.array(tmp_rmsds)))

    return np.array(lowest_rmsd)

# class BasicMolecularMetrics(object):
#     def __init__(self, dataset_info, dataset_smiles_list=None):
#         self.atom_decoder = dataset_info['atom_decoder']
#         self.dataset_smiles_list = dataset_smiles_list
#         self.dataset_info = dataset_info

#         # Retrieve dataset smiles only for qm9 currently.
#         # if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
#         #     self.dataset_smiles_list = retrieve_qm9_smiles(
#         #         self.dataset_info)


#     def compute_validity(self, generated):
#         """ generated: list of couples (positions, atom_types)"""
#         valid = []
#         complete_n = 0
#         for graph in generated:
#             mol = build_molecule(*graph, self.dataset_info)
#             smiles = mol2smiles(mol)
#             if smiles is not None:
#                 try:
#                     mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
#                 except:
#                     continue
#                 if len(mol_frags) == 1:
#                     complete_n += 1

#                 largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
#                 smiles = mol2smiles(largest_mol)
#                 valid.append(smiles)

#         return valid, len(valid) / len(generated), complete_n / len(generated)

#     def compute_uniqueness(self, valid, generated):
#         """ valid: list of SMILES strings."""
#         return list(set(valid)), len(set(valid)) / len(generated)

#     def compute_novelty(self, unique, generated):
#         num_novel = 0
#         novel = []
#         for smiles in unique:
#             if smiles not in self.dataset_smiles_list:
#                 novel.append(smiles)
#                 num_novel += 1
#         return novel, num_novel / len(generated)

#     def evaluate(self, generated):
#         """ generated: list of pairs (positions: n x 3, atom_types: n [int])
#             the positions and atom types should already be masked. """
#         valid, validity, complete = self.compute_validity(generated)
#         print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
#         if validity > 0:
#             unique, uniqueness = self.compute_uniqueness(valid, generated)
#             print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

#             if self.dataset_smiles_list is not None:
#                 _, novelty = self.compute_novelty(unique, generated)
#                 print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
#             else:
#                 novelty = 0.0
#         else:
#             novelty = 0.0
#             uniqueness = 0.0
#             unique = None

            
#         return dict(Validity=validity, Unique=uniqueness, Novelty=novelty, Complete=complete)



