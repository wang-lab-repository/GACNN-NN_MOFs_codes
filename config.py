
import torch
class Config():
    element_electronegativity = {  #元素电负性对照表
    'H': 3.04,  
    'Li': 2.17,  
    'Na': 2.15,  
    'K': 2.07,  
    'Rb': 2.07,  
    'Cs': 1.97,  
    'Fr': 2.01,  
    'Be': 2.42,  
    'Mg': 2.39,  
    'Ca': 2.2,  
    'Sr': 2.13,  
    'Ba': 2.02,  
    'Sc': 2.35,  
    'Ti': 2.23,  
    'V': 2.08,  
    'Cr': 2.12,  
    'Mn': 2.2,  
    'Fe': 2.32, 
    'Co': 2.34,
    'Ni': 2.32,
    'Cu': 2.86,
    'Zn': 2.26,
    'Y': 2.52,
    'Zr': 2.05,
    'Nb': 2.59,
    'Mo': 2.47,
    'Tc': 2.82,
    'Ru': 2.68,
    'Rh': 2.65,
    'Pd': 2.7,
    'Ag': 2.88,
    'Cd': 2.36,
    'Hf': 2.01,
    'Ta': 2.32,
    'W': 2.42,
    'Re': 2.59,
    'Os': 2.72,
    'Ir': 2.79,
    'Pt': 2.98,
    'Au': 2.81,
    'Hg': 2.92,
    'B': 3.04,
    'Al': 2.52,
    'Ga': 2.43,
    'In': 2.29,
    'Tl': 2.26,
    'C': 3.15,
    'Si': 2.82,
    'Ge': 2.79,
    'Sn': 2.68,
    'Pb': 2.62,
    'N': 3.56,
    'P': 3.16,
    'As': 3.15,
    "Sb": 3.05,
    'O': 3.78,
    'S':3.44,
    'Se': 3.37,
    'Te': 3.14,
    'F': 4.0,
    'Cl': 3.56,
    'Br': 3.45,
    'I': 3.2,
    'La': 2.49,
    'Ce': 2.61,
    'Pr': 2.24,
    'Nd': 2.11,
    'Sm': 1.9,
    'Eu': 1.81,
    'Gd': 2.4,
    'Tb': 2.29,
    'Dy': 2.07,
    'Ho': 2.12,
    'Er': 2.02,
    'Tm': 2.03,
    'Tb': 1.78,
    'Lu': 2.68,
    'Th': 2.62,
    'U': 2.45
    
}
    bond_type = {
    'nc': 0.0,
    'un': 0.0,
    'du': 0.5,
    'am': 1.0,
    '1': 1.0,
    'ar': 1.5,
    '2': 2.0,
    '3': 3.0   
    }
    # file path
    excel_path = './MOF_.xlsx'
    graph_file_dir = './dataset'
    # Model structure parameters
    graph_input_dim = 4  # 图节点的特征维度，例如：10
    graph_hidden_dim = 64  # 隐藏层维度
    graph_output_dim = 128  # 输出图特征维度
    graph_num_heads = 4
    # smiles_features_dim = 167
    smiles_features_dim = 1024
    fc_input_dim = 28 + smiles_features_dim  # 特征维度：数值特征 
    fc_hidden_dim = 64  # 全连接层的隐藏维度
    fc_output_dim = 32  # 全连接层输出维度2-20原实验所用参数
    # fc_hidden_dim = 512
    # fc_output_dim = 128
    embed_dim = 128
    num_heads = 4
    # parameters of training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1024
    shuffle = True
    epochs = 4000
    lr = 0.0010
    random_state = 165
