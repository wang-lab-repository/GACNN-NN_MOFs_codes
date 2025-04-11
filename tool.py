import numpy as np
import networkx as nx
from ase import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.metrics import r2_score,mean_absolute_percentage_error
import dgl
import torch
import os
from rdkit import Chem
from rdkit.Chem import AllChem,MACCSkeys,rdMolDescriptors,rdFingerprintGenerator
from config import Config
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
opt = Config()
def parse_mol2(mol2_file):
    atoms = []
    bonds = []
    atoms_type_list = []
    with open(mol2_file, 'r') as file:
        lines = file.readlines()
        in_atoms = False
        in_bonds = False

        for line in lines:
            if line.startswith('@<TRIPOS>ATOM'):
                in_atoms = True
                continue
            elif line.startswith('@<TRIPOS>BOND'):
                in_atoms = False
                in_bonds = True
                continue
            elif line.startswith('@<TRIPOS>SUBSTRUCTURE'):
                in_atoms = False
                in_bonds = False
                break
            if in_atoms:
                parts = line.split()
                # print(parts)
                atom_id = int(parts[0])      # 原子编号
                atom_name = parts[1]         # 原子名称
                
                x = float(parts[2])           # x坐标
                y = float(parts[3])           # y坐标
                z = float(parts[4])           # z坐标
                atom_type = parts[5].split('.')[0]         # 原子类型
                partial_charge = float(parts[-1])  # 部分电荷
                atoms.append((atom_id, atom_name, atom_type, (x, y, z), partial_charge))
                atoms_type_list.append(atom_type)
            if in_bonds:
                parts = line.split()
                bond_id = int(parts[0])      # 键编号
                atom1 = int(parts[1]) -1     # 连接的第一个原子（0索引）
                atom2 = int(parts[2]) -1    # 连接的第二个原子（0索引）
                bond_type = parts[3]         # 键的类型（单、双、三等）
                bonds.append((atom1, atom2, bond_type))

    return atoms, bonds,atoms_type_list



def create_graph(atoms, bonds):
    num_atoms = len(atoms)
    graph = dgl.graph(([], []), num_nodes=num_atoms)

    # 添加边
    src, dst = zip(*[(bond[0], bond[1]) for bond in bonds])
    graph.add_edges(src, dst)

    # 设置节点特征
    atom_features = []
    
    for atom in atoms:
        atom_id, atom_name, atom_type, coords, partial_charge = atom
        # feature = [
        #     atom_type,                # 原子类型
        #     partial_charge,          # 部分电荷
        #     *coords                  # 坐标
        # ]
        feature = [
                         # 原子类型
            opt.element_electronegativity[atom_type],          # 电负性
            *coords                  # 坐标
        ]
        
        atom_features.append(feature)
        
    # 将特征转换为张量
    atom_features = torch.tensor(atom_features, dtype=torch.float)

    # 将特征赋给图的节点
    graph.ndata['feat'] = atom_features
    
    # 设置边特征
    bond_features = []
    for bond in bonds:
        atom1 = atoms[bond[0]]
        atom2 = atoms[bond[1]]
        
        bond_type = bond[2]  # 键类型

        # 将边特征组合在一起
        # feature = [
        #     bond_type,          # 键类型
        #     bond_length         # 键长度
        # ]
        feature = [
                     # 键类型
            opt.bond_type[bond_type]        # 键长度
        ]
        bond_features.append(feature)

    # 将边特征转换为张量
    bond_features = torch.tensor(bond_features, dtype=torch.float)

    # 将特征赋给图的边
    graph.edata['feat'] = bond_features

    return graph

def smiles_to_morgan_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=nBits)
        fp = generator.GetFingerprintAsNumPy(mol=mol)
        # fp = generator.GetCountFingerprintAsNumPy(mol=mol)
        return fp
        
    else:
        return np.zeros(nBits)

def smiles_to_maccs_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return rdMolDescriptors.GetMACCSKeysFingerprint(mol=mol)
        # return MACCSkeys.GenMACCSKeys(mol)
    else:
        return np.zeros(167)  # MACCS Keys有167个位

def smiles_to_rdkit_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nBits)
        fp = generator.GetFingerprintAsNumPy(mol=mol)
        return fp
    else:
        rdFingerprintGenerator.GetTopologicalTorsionGenerator
        return np.zeros(nBits)  

def smiles_to_topological_fingerprint(smiles, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nBits)
        fp = generator.GetFingerprintAsNumPy(mol=mol)
        return fp
    else:
        
        return np.zeros(nBits)  

def smiles_to_atom_pair_fingerprint(smiles, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nBits)
        fp = generator.GetFingerprintAsNumPy(mol=mol)
        return fp
    else:
        return np.zeros(nBits)

def collate_fn(batch):
    # 从batch中提取DGLGraph对象，inputs, targets等
        graphs = [item[0] for item in batch]  # 假设每个元素是 (graph, input, target, atom_list)
        inputs = [item[1] for item in batch]
        targets = [item[2] for item in batch]
        atom_lists = [item[3] for item in batch]
        mof_indexs = [item[4] for item in batch]
        # 创建节点索引，保证每个节点与其原子类型一一对应
        node_indices = []
        for i, graph in enumerate(graphs):
            # 生成当前图的节点索引，并加上偏移量
            node_indices_i = torch.arange(graph.num_nodes()) + i * graph.num_nodes()
    
            # 使用 append 将生成的索引添加到列表中
            node_indices.append(node_indices_i)

        # 使用 torch.cat 将列表中的所有 Tensor 按维度 0 拼接在一起
        node_indices = torch.cat(node_indices, dim=0)
        # 将 DGLGraph 对象合并成一个批次，使用 DGL 的功能
        # batched_graph = dgl.batch(graphs)
        inputs = torch.stack([input_item.clone().detach() if isinstance(input_item, torch.Tensor) else torch.tensor(input_item) for input_item in inputs], dim=0)
        targets = torch.stack([target_item.clone().detach() if isinstance(target_item, torch.Tensor) else torch.tensor(target_item) for target_item in targets], dim=0)
        graphs = dgl.batch(graphs)

        # atom_lists 保留原样，或者根据需求转换
        # atom_lists = [torch.tensor(atom_item) for atom_item in atom_lists]
        # return batched_graph, inputs, targets, atom_lists
        return graphs, inputs, targets, atom_lists, node_indices, mof_indexs
def model_predict(model,data):
    return model(data)

def visualize_graph(graph,atoms_type):
    fig = go.Figure()
    # ax = fig.add_subplot(111, projection='3d')
    coords = graph.ndata['feat'][:, 1:4].numpy()  # 假设坐标在第1到3列
    # 绘制节点
    color_map = {
        'H': 'white',   # 氢
        'C': 'gray',    # 碳
        'O': 'red',     # 氧
        'N': 'blue',    # 氮
        'S': 'yellow'   # 硫
        # 添加更多元素类型和对应颜色
    }
    
    # 为每个原子分配颜色
    colors = [color_map.get(atom, 'black') for atom in atoms_type]
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(size=5, color=colors, line=dict(width=1)),
        text=[f'Atom: {atom}' for atom in atoms_type],  # 显示原子类型
        hoverinfo='text'
    ))
    # print(graph.edges())
    # 绘制边
    [bond_front, bond_rear] = graph.edges()
    for i in range(len(bond_front)):
        
        atom1_id = bond_front[i]
        atom2_id = bond_rear[i]
        atom1_coords = coords[atom1_id]
        atom2_coords = coords[atom2_id]
        fig.add_trace(go.Scatter3d(
            x=[atom1_coords[0], atom2_coords[0]],
            y=[atom1_coords[1], atom2_coords[1]],
            z=[atom1_coords[2], atom2_coords[2]],
            mode='lines',
            line=dict(color='black', width=2)
        ))
    # 设置标签
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        title='3D Molecular Structure',
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()

# def visualize_graph_bymodel(graph, atoms_type, node_importance):
#     fig = go.Figure()

#     coords = graph.ndata['feat'][:, 1:4].detach().numpy()  # 假设坐标在第1到3列

#     # 为每个原子分配颜色
#     color_map = {
#         'H': 'white',   # 氢
#         'C': 'gray',    # 碳
#         'O': 'red',     # 氧
#         'N': 'blue',    # 氮
#         'S': 'yellow'   # 硫
#     }

#     # 为每个原子分配颜色（原子类型）
#     colors = [color_map.get(atom, 'black') for atom in atoms_type]

#     # 归一化节点重要性
#     norm_node_importance = mcolors.Normalize(vmin=np.min(node_importance), vmax=np.max(node_importance))

#     # 定义颜色映射：正值 -> 蓝色到白色，负值 -> 红色到白色
#     def get_node_color(importance):
#         # 归一化到0到1之间
#         norm = (importance - np.min(node_importance)) / (np.max(node_importance) - np.min(node_importance))
#         if importance >= 0:
#             return plt.cm.Blues(norm)  # 正重要性：蓝色渐变
#         else:
#             return plt.cm.Reds(-norm)  # 负重要性：红色渐变

#     node_colors = [get_node_color(importance) for importance in node_importance]

#     # 创建原子信息：显示原子类型和对应的重要性值
#     node_text = [f'Atom: {atom}<br>Importance: {importance:.2f}' 
#                 for atom, importance in zip(atoms_type, node_importance)]

#     # 绘制节点，使用节点重要性影响颜色
#     fig.add_trace(go.Scatter3d(
#         x=coords[:, 0],
#         y=coords[:, 1],
#         z=coords[:, 2],
#         mode='markers',
#         marker=dict(size=8, color=node_colors, line=dict(width=1), opacity=0.8),  # 节点的透明度控制
#         text=node_text,  # 显示原子类型和重要性
#         hoverinfo='text'  # 显示文本信息
#     ))

#     # 添加薄雾效果
#     # 获取每个节点的薄雾颜色（根据其重要性）
#     def get_fog_color(importance):
#         norm = (importance - np.min(node_importance)) / (np.max(node_importance) - np.min(node_importance))
#         if importance >= 0:
#             return plt.cm.Blues(norm)  # 正重要性：蓝色薄雾
#         else:
#             return plt.cm.Reds(-norm)  # 负重要性：红色薄雾

#     for i in range(len(coords)):
#         fog_color = get_fog_color(node_importance[i])[:3]  # 取RGB颜色部分
#         fig.add_trace(go.Scatter3d(
#             x=[coords[i, 0]] * 2, 
#             y=[coords[i, 1]] * 2, 
#             z=[coords[i, 2]] * 2,
#             mode='markers',
#             marker=dict(size=10, color=fog_color, opacity=0.2),  # 大小和透明度控制薄雾效果
#             showlegend=False
#         ))

#     # 获取边的列表
#     [bond_front, bond_rear] = graph.edges()

    

#     edge_features = graph.edata['feat']

#     # 选择边的一个特征值（例如，第一个特征）来决定边的粗细
#     edge_weights = edge_features[:, 0].detach().numpy()  # 你可以根据需要选择不同的特征
    
#     # 绘制边，使用边的特征值影响线条宽度
#     for i in range(len(bond_front)):
#         atom1_id = bond_front[i].item()
#         atom2_id = bond_rear[i].item()
#         atom1_coords = coords[atom1_id]
#         atom2_coords = coords[atom2_id]

#         # 根据边的特征值设置线条宽度
#         line_width = max(2 * edge_weights[i], 0.1)
#         fig.add_trace(go.Scatter3d(
#             x=[atom1_coords[0], atom2_coords[0]],
#             y=[atom1_coords[1], atom2_coords[1]],
#             z=[atom1_coords[2], atom2_coords[2]],
#             mode='lines',
#             line=dict(color='black', width=line_width),  # 使用颜色和宽度
#             # text=edge_weights[i],  # 显示边的重要性
#             hoverinfo='text'  # 显示边的文本信息
#         ))

    

#     # 设置标签和布局
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#             aspectmode='cube'
#         ),
#         title='3D Molecular Structure with Fog and Gradient Effects',
#         margin=dict(l=0, r=0, b=0, t=50)
#     )

#     fig.show()
def visualize_graph_bymodel(graph, atoms_type, node_importance):
    fig = go.Figure()

    coords = graph.ndata['feat'][:, 1:4].detach().numpy()  # 假设坐标在第1到3列

    # 定义原子类型颜色映射
    color_map = {
        'H': 'white',   # 氢
        'C': 'gray',    # 碳
        'O': 'red',     # 氧
        'N': 'blue',    # 氮
        'S': 'yellow',  # 硫
        'Zn': 'purple', # 锌
        'Fe': 'magenta', # 铁
        'metal': 'orange'  # 其他金属原子
    }

    # 定义原子类型大小映射
    size_map = {
        'H': 4,   # 氢
        'C': 8,   # 碳
        'O': 10,  # 氧
        'N': 8,   # 氮
        'S': 12,  # 硫
        'Zn': 14, # 锌
        'Fe': 14, # 铁
        'metal': 12  # 其他金属原子
    }

    # 从 atoms_type 的元组中提取原子符号
    atoms = [atom[0] for atom in atoms_type]

    # 为每个原子分配颜色和大小
    colors = [color_map.get(atom, 'black') for atom in atoms]
    sizes = [size_map.get(atom, 8) for atom in atoms]

    # 绘制节点，使用原子类型颜色和大小
    added_legends = set()  # 用于记录已经添加到图例的原子类型
    for i, (atom, color, size) in enumerate(zip(atoms, colors, sizes)):
        if atom not in added_legends:
            fig.add_trace(go.Scatter3d(
                x=[coords[i, 0]],
                y=[coords[i, 1]],
                z=[coords[i, 2]],
                mode='markers',
                marker=dict(size=size, color=color, line=dict(width=1), opacity=0.8),
                text=f'Atom: {atoms_type[i]}',
                hoverinfo='text',
                legendgroup=atom,
                name=atom,
                showlegend=True
            ))
            added_legends.add(atom)
        else:
            fig.add_trace(go.Scatter3d(
                x=[coords[i, 0]],
                y=[coords[i, 1]],
                z=[coords[i, 2]],
                mode='markers',
                marker=dict(size=size, color=color, line=dict(width=1), opacity=0.8),
                text=f'Atom: {atoms_type[i]}',
                hoverinfo='text',
                legendgroup=atom,
                showlegend=False
            ))

    # 标注特征重要性为正的前10%和为负的后10%的节点
    threshold_positive = np.percentile(node_importance, 90)
    threshold_negative = np.percentile(node_importance, 10)

    for i, (atom, importance) in enumerate(zip(atoms, node_importance)):
        if importance > threshold_positive or importance < threshold_negative:
            text_color = 'red' if importance > 0 else 'green'
            fig.add_trace(go.Scatter3d(
                x=[coords[i, 0]],
                y=[coords[i, 1]],
                z=[coords[i, 2]],
                mode='text',
                text=f'{atom}',
                textposition="bottom center",
                textfont=dict(color=text_color),
                showlegend=False
            ))

    # 添加图例条目以显示正重要性和负重要性的颜色
    fig.add_trace(go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Positive Importance',
        showlegend=True
    ))

    fig.add_trace(go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Negative Importance',
        showlegend=True
    ))

    # 获取边的列表
    [bond_front, bond_rear] = graph.edges()
    edge_features = graph.edata['feat']

    # 选择边的一个特征值（例如，第一个特征）来决定边的粗细
    edge_weights = edge_features[:, 0].detach().numpy()  # 你可以根据需要选择不同的特征

    # 绘制边，使用边的特征值影响线条宽度
    for i in range(len(bond_front)):
        atom1_id = bond_front[i].item()
        atom2_id = bond_rear[i].item()
        atom1_coords = coords[atom1_id]
        atom2_coords = coords[atom2_id]

        # 根据边的特征值设置线条宽度
        line_width = max(2 * edge_weights[i], 0.1)
        fig.add_trace(go.Scatter3d(
            x=[atom1_coords[0], atom2_coords[0]],
            y=[atom1_coords[1], atom2_coords[1]],
            z=[atom1_coords[2], atom2_coords[2]],
            mode='lines',
            line=dict(color='black', width=line_width),
            hoverinfo='none',
            showlegend=False
        ))

    # 设置标签和布局
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        title='3D Molecular Structure with Highlighted Important Nodes',
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(title='Legend')
    )

    fig.show()
def draw_loss(loss_list):
    x = np.arange(len(loss_list))
    # x = len(train_acc_list)
    plt.plot(x, loss_list, label='train loss')
    
    plt.xlabel("epochs")
    plt.ylabel("loss")  
    plt.savefig('result/the_total_train_loss.png') 
    pass

def get_R2_RMSE(model,dataloader,device):
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradient computation for inference
        for graph, input_tensor, target_tensor, atoms_type ,   mof_indexs, smiles in dataloader:
            
            
            # Forward pass
            outputs,_,_ = model(graph,  input_tensor,atoms_type)
            
            # Store predictions and labels
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(target_tensor.cpu().numpy())
    
    # Flatten the lists into one array
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute R² score
    r2 = r2_score(all_labels, all_preds)

    # Compute RMSE
    rmse = np.sqrt(((all_labels - all_preds) ** 2).mean())
    mape = mean_absolute_percentage_error(all_labels,all_preds)
    return r2, rmse, mape

if __name__ == '__main__':
    # 示例：使用上述函数
    mol2_file = 'dataset/2336490.mol2'
    atoms, bonds, atoms_type_list = parse_mol2(mol2_file)
    graph = create_graph(atoms, bonds)

    print("图数据转换完成")
    print(graph)

    # visualize_graph(graph, atoms_type_list)
    

