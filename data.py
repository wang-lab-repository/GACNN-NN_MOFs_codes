import os 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
from config import Config
from tool import parse_mol2,create_graph,collate_fn,smiles_to_atom_pair_fingerprint,smiles_to_maccs_fingerprint,smiles_to_morgan_fingerprint,smiles_to_rdkit_fingerprint,smiles_to_topological_fingerprint

class MultiModalDataset(Dataset):
    def __init__(self,df,graph_file_dir) -> None:
        super(MultiModalDataset,self).__init__()
        self.df = df
        self.graph_file_dir = graph_file_dir
        self.features_columns = ['Al', 'Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Ni', 'Zn', 'Zr',  
                                 'initial concentration of adsorbates(ppm)', 'initial concentration of MOF(mg/ml)', 
                                 'Metal loading(wt.%)', 'Sonochemical', 'Microwave', 'mechanochemical', 'hydrothermal', 
                                 'Solvothermal', 'Monoclinic', 'Tetragonal', 'orthorhombic', 'trigonal', 'cubic', 'Triclinic', 'hexagonal', 'time(min)', 'PH', 'temperature']
        self.target_column = 'Adsorption(mg/g)'
        self.smiles_column = 'Adsorbates'
        pass
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        graph_file_index = self.df.loc[index, 'mof_id']
        graph_file_path = os.path.join(self.graph_file_dir, str(graph_file_index) + '.mol2')
        atoms, bonds, atoms_type_list = parse_mol2(graph_file_path)
        graph = create_graph(atoms, bonds)
        features = self.df.loc[index, self.features_columns].values.astype(np.float32)

        
        target = self.df.loc[index, self.target_column].astype(np.float32)

        # 获取 SMILES 字符串，并转化为特征（Morgan Fingerprint）
        smiles = self.df.loc[index, self.smiles_column]
        # smiles_features = smiles_to_maccs_fingerprint(smiles)
        
        # smiles_features = smiles_to_atom_pair_fingerprint(smiles=smiles)
        # smiles_features = smiles_to_topological_fingerprint(smiles=smiles)
        smiles_features = smiles_to_morgan_fingerprint(smiles=smiles)
        # smiles_features = smiles_to_rdkit_fingerprint(smiles=smiles)
        # 将特征合并成最终的输入向量
        # 这里假设你希望将数值特征和 SMILES 特征拼接起来作为输入
        # print("the shape of figerprint is:", len(smiles_features))
        input_features = np.concatenate([features, smiles_features])

        # 将输入特征和目标值转换为 PyTorch 张量
        input_tensor = torch.tensor(input_features, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        
        return graph, input_tensor, target_tensor, atoms_type_list, graph_file_index, smiles
    

if __name__ == '__main__':
    opt = Config()
    dataset = MultiModalDataset(df=pd.read_excel(opt.excel_path),graph_file_dir=opt.graph_file_dir)
    dataloader = dgl.dataloading.GraphDataLoader(dataset=dataset,batch_size=opt.batch_size)
    # dataloader = DataLoader(dataset=dataset,batch_size=opt.batch_size,shuffle=opt.shuffle,collate_fn=collate_fn)
    for graph, inputs, targets, atoms_lists, graph_file_index in dataloader:
        print(graph)
        print("------")
        print("the shape of inputs is:",inputs.shape)
        print(inputs)
        print("the shape of targets is:", targets.shape)
        print(targets)
        print("the shape of atoms_lists is:", len(atoms_lists))
        # print(atoms_lists)
        break