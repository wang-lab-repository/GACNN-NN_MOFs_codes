import torch 
import torch.nn as nn
import torch.optim as optim

import dgl

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from captum.attr import IntegratedGradients
from tool import collate_fn,model_predict,draw_loss,get_R2_RMSE,visualize_graph_bymodel
from data import MultiModalDataset
from model import MultiModalModel
from config import Config
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib import colors as mcolors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem,MACCSkeys,rdMolDescriptors,rdFingerprintGenerator
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import AllChem
from tool import parse_mol2,create_graph,smiles_to_morgan_fingerprint
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Geometry import Point2D


def visualize_molecule(graph, atoms_type_graph, node_importance_graph, smiles, node_importance_smiles):
    # Define atom type color and size mapping
    color_map = {
        'H': 'gray',
        'C': 'lightgray',
        'O': 'tomato',
        'N': 'skyblue',
        'S': 'goldenrod',
        'Zn': 'mediumorchid',
        'Fe': 'magenta',
        'other metal': 'sienna'
    }

    size_map = {
        'H': 4,
        'C': 8,
        'O': 10,
        'N': 8,
        'S': 12,
        'Zn': 14,
        'Fe': 14,
        'metal': 14
    }

    def visualize_graph(fig, row, col, atoms_type, node_importance, title):
        coords = graph.ndata['feat'][:, 1:4].detach().numpy()
        
        atoms = [atom[0] for atom in atoms_type]
        colors = [color_map.get(atom, 'sienna') for atom in atoms]
        sizes = [size_map.get(atom, 8) for atom in atoms]

        # Draw nodes
        for i, (atom, color, size) in enumerate(zip(atoms, colors, sizes)):
            # Extract electronegativity from node features (last element)
            electronegativity = graph.ndata['feat'][i, 0].item()
            importance = node_importance[i]
            hover_text = f"{atom}<br>Electronegativity: {electronegativity:.2f}<br>Importance: {importance:.2f}"

            fig.add_trace(
                go.Scatter3d(
                    x=[coords[i, 0]],
                    y=[coords[i, 1]],
                    z=[coords[i, 2]],
                    mode='markers',
                    legendgroup=atom,
                    marker=dict(size=size, color=color, line=dict(width=1), opacity=0.8),
                    showlegend=True if i == 0 else False,
                    name=atom,  # Assign a name for the legend
                    textfont=dict(family='Arial', size=16),
                    hoverinfo='text',
                    text=hover_text
                ),
                row=row,
                col=col
            )

        # Calculate the number of atoms to label
        num_atoms = len(atoms)
        max_to_label = int(0.10 * num_atoms)  # 10% of total atoms
        # threshold_positive = np.percentile(node_importance, 99.9)
        # threshold_negative = np.percentile(node_importance, 0.1)

        # positive_importance = np.sort(node_importance)[::-1]
        # negative_importance = np.sort(node_importance)

        # Get indices to label
        pos_indices = np.argsort(node_importance)[::-1][:max_to_label//2]  # Top 50% positive
        neg_indices = np.argsort(node_importance)[:max_to_label//2]  # Top 50% negative

        # Combine and deduplicate indices
        label_indices = np.unique(np.concatenate([pos_indices, neg_indices]))

        # Label importance
        for i in label_indices:
            atom = atoms[i]
            importance = node_importance[i]
            text_color = 'red' if importance > 0 else 'blue'
            fig.add_trace(
                go.Scatter3d(
                    x=[coords[i, 0]],
                    y=[coords[i, 1]],
                    z=[coords[i, 2]],
                    mode='text',
                    text=f'{atom} ',
                    textposition='top center',  # Adjust text position
                    textfont=dict(family='Arial', color=text_color, size=12),  # Smaller font size
                    showlegend=False
                ),
                row=row,
                col=col
            )

        # Draw edges
        bond_front, bond_rear = graph.edges()
        edge_features = graph.edata['feat']
        edge_weights = edge_features[:, 0].detach().numpy()

        for i in range(len(bond_front)):
            atom1_id = bond_front[i].item()
            atom2_id = bond_rear[i].item()
            atom1_coords = coords[atom1_id]
            atom2_coords = coords[atom2_id]

            line_width = max(2 * edge_weights[i], 0.1)
            fig.add_trace(
                go.Scatter3d(
                    x=[atom1_coords[0], atom2_coords[0]],
                    y=[atom1_coords[1], atom2_coords[1]],
                    z=[atom1_coords[2], atom2_coords[2]],
                    mode='lines',
                    line=dict(color='black', width=line_width),
                    showlegend=False
                ),
                row=row,
                col=col
            )

        # Set axes and title
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                xaxis_title_font=dict(family='Arial', size=14),
                yaxis_title_font=dict(family='Arial', size=14),
                zaxis_title_font=dict(family='Arial', size=14)
            )
        )

    def visualize_smiles(fig, row, col, smiles, feature_importance, title):
        mol_graph = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol_graph)
        mol = mol_graph

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = [mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())]

        # Calculate atom importance
        atom_importances = np.zeros(mol.GetNumAtoms())
    
        # Ensure feature_importance is a 1D array
        feature_importance = feature_importance.flatten()  # Convert (1, 1024) to (1024, )

        # Calculate importance for each atom
        for atom_idx in range(mol.GetNumAtoms()):
            atom_fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(
                mol=mol, fromAtoms=[atom_idx]
            )
            for bit in atom_fingerprint.GetOnBits():
                if bit < len(feature_importance):
                    atom_importances[atom_idx] += feature_importance[bit]

        # Automatically generate coordinate range
        all_x = [pos.x for pos in coords]
        all_y = [pos.y for pos in coords]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Remove dots and directly label importance values
        threshold_positive = np.percentile(atom_importances, 80)
        threshold_negative = np.percentile(atom_importances, 20)
        mask = (atom_importances > threshold_positive) | (atom_importances < threshold_negative)

        # Initialize a list to track which atoms have been labeled
        labeled = [False] * mol.GetNumAtoms()

        # Label atoms based on importance
        for i in np.where(mask)[0]:
            if atoms[i] not in ['C', 'H']:
                labeled[i] = True
                text = f"{atoms[i]} "
                text_color = 'red' if atom_importances[i] > 0 else 'blue'
                fig.add_trace(
                    go.Scatter(
                        x=[coords[i].x],
                        y=[coords[i].y],
                        mode='text',
                        text=text,
                        textposition='top center',  # Adjust text position
                        textfont=dict(family='Arial', color=text_color, size=12),  # Smaller font size
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )

        # Label other atoms (non C/H) if not labeled by importance
        for i in range(mol.GetNumAtoms()):
            if not labeled[i] and atoms[i] not in ['C', 'H']:
                fig.add_trace(
                    go.Scatter(
                        x=[coords[i].x],
                        y=[coords[i].y],
                        mode='text',
                        text=atoms[i],
                        textposition='bottom center',
                        textfont=dict(family='Arial', color='black', size=12),  # Smaller font size
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )

        # Redraw the molecular structure
        for bond in mol.GetBonds():
            atom1_id = bond.GetBeginAtomIdx()
            atom2_id = bond.GetEndAtomIdx()
            atom1_coords = coords[atom1_id]
            atom2_coords = coords[atom2_id]

            bond_type = bond.GetBondType()
            dash_styles = {
                Chem.BondType.SINGLE: 'solid',
                Chem.BondType.DOUBLE: 'dash',
                Chem.BondType.TRIPLE: 'dot',
                Chem.BondType.AROMATIC: 'dashdot'
            }

            line_dash = dash_styles.get(bond_type, 'solid')
            fig.add_trace(
                go.Scatter(
                    x=[atom1_coords.x, atom2_coords.x],
                    y=[atom1_coords.y, atom2_coords.y],
                    mode='lines',
                    line=dict(color='black', width=1, dash=line_dash),
                    showlegend=False
                ),
                row=row,
                col=col
            )

        # Update axes range
        fig.update_xaxes(
            range=[x_min - 1, x_max + 1],
            title_text='X',
            title_font=dict(family='Arial', size=14),
            row=row,
            col=col
        )
        fig.update_yaxes(
            range=[y_min - 1, y_max + 1],
            title_text='Y',
            title_font=dict(family='Arial', size=14),
            row=row,
            col=col
        )

    # Create subplot layout
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("Graph Visualization", "SMILES Visualization"),
        column_widths=[0.6, 0.4]
    )

    # Adjust subplot titles
    for i, title in enumerate(fig.layout.annotations):
        title.update(
            text=title.text,
            font=dict(family='Arial', size=16),
            xshift=-40 if i == 0 else 40  # Adjust positions
        )

    # Call plotting functions
    visualize_graph(fig, 1, 1, atoms_type_graph, node_importance_graph, "Graph Visualization")
    visualize_smiles(fig, 1, 2, smiles, node_importance_smiles, "SMILES Visualization")

    # Add global legend
    for atom, color in color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                name=atom,
                marker=dict(color=color, size=10),
                showlegend=True,
                textfont=dict(family='Arial', size=16)
            ),
            row=1,
            col=2
        )

    # Add legend for positive and negative importance
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='text',
            name='Positive Importance',
            text=['Positive Importance'],
            textfont=dict(family='Arial', color='red', size=16),
            showlegend=True
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='text',
            name='Negative Importance',
            text=['Negative Importance'],
            textfont=dict(family='Arial', color='blue', size=16),
            showlegend=True
        ),
        row=1,
        col=2
    )

    # Add legend for bond types
    bond_styles = [
        ('Single Bond', 'solid'),
        ('Double Bond', 'dash'),
        ('Triple Bond', 'dot'),
        ('Aromatic Bond', 'dashdot')
    ]

    for name, dash in bond_styles:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name=name,
                line=dict(color='black', width=1, dash=dash),
                showlegend=True
            ),
            row=1,
            col=2
        )

    # Scale the SMILES visualization axes
    fig.update_xaxes(range=[-15, 15], row=1, col=2)
    fig.update_yaxes(range=[-15, 15], row=1, col=2)

    # Display the figure
    fig.show()


def vision():
    #可视化的代码
    opt = Config()
    # df = pd.read_excel(opt.excel_path)
    # # subdf = df.iloc[792].reset_index(drop=True)
    # dataset = MultiModalDataset(df=df,graph_file_dir=opt.graph_file_dir)
    model = MultiModalModel(graph_input_dim=opt.graph_input_dim,graph_hidden_dim=opt.graph_hidden_dim,graph_output_dim=opt.graph_output_dim,
                            fc_input_dim=opt.fc_input_dim,fc_hidden_dim=opt.fc_hidden_dim,fc_output_dim=opt.fc_output_dim
                            )
    model.load_state_dict(torch.load('checkpoints/2-19-18-32GACNN_NN_mulattfusion.pth'))
    model.train()
    # date = dataset[0]
    file,feature_tensor,smiles,target_tensor#假设传入的是这四个
    smiles_feature = smiles_to_morgan_fingerprint(smiles=smiles)
    input_features = torch.cat([feature_tensor, smiles_feature], dim=-1)
    atoms, bonds,atoms_type=parse_mol2(file)
    graph=create_graph(atoms=atoms,bonds=bonds)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    
    input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
    # batch_graph = dgl.batch([graph])  
    graph.ndata['feat'].requires_grad_(True)
        # graph.edata['feat'].requires_grad_(True)
        # # print(graph.ndata['feat'].grad)
    input_tensor.requires_grad_()
        
        
        # grad_of_feature = input_tensor.grad
       
        
        
        # other_importance_list.append(other_importance)
        # feature_importance = grad_of_feature.detach().numpy()[:, -1024:]
        # feature_importance_list.append(feature_importance)
    optimizer.zero_grad()  # 清空梯度
    prediction, _, _ = model(graph, input_tensor, atoms_type)
    loss = criterion(prediction.view(-1), target_tensor)
    loss.backward()  # 反向传播
        
        # print("the shape of graph is :",graph.ndata['feat'].grad.shape)
    grad_of_node = graph.ndata['feat'].grad[:,0]
    node_importance = grad_of_node.detach().numpy()
        # node_importance_list.append(node_importance)
    optimizer.step()  # 更新权重
    # print(graph.ndata['feat'].grad)
    # print(graph.edata['feat'].grad)
    weights = model.state_dict()
    feature_importance = weights['nn.fc1.weight'].sum(dim=0).detach().numpy()[ -1024:]
    # print(input_weights.shape)
    
    
    # other_importance_np = np.array(other_importance_list)
    # feature_importance_np = np.array(feature_importance_list)
    # grad_of_edge = graph.edata['feat'].grad[:,-1]
    
    
    # print("the shape of node is:",node_importance_np.shape)
    # print(graph_file_index)
    smiles = smiles
    # print(type(smiles))
    # node_importance = np.mean(node_importance_np, axis=0)
    # edge_importance = grad_of_edge.detach().numpy()
    # node_importance = np.mean(node_importance_np, axis=0)
    # other_importance = weights['nn.fc1.weight'].sum(dim=0).detach().numpy()[0:26]

    visualize_molecule(graph=graph,atoms_type_graph=atoms_type,node_importance_graph=node_importance,
                       smiles=smiles,node_importance_smiles=feature_importance)
    
def prediction():
    opt = Config()
    
    model = MultiModalModel(graph_input_dim=opt.graph_input_dim,graph_hidden_dim=opt.graph_hidden_dim,graph_output_dim=opt.graph_output_dim,
                            fc_input_dim=opt.fc_input_dim,fc_hidden_dim=opt.fc_hidden_dim,fc_output_dim=opt.fc_output_dim
                            )
    model.load_state_dict(torch.load('checkpoints/2-19-18-32GACNN_NN_mulattfusion.pth'))
    file,feature_tensor,smiles#假设传入的是这三个
    smiles_feature = smiles_to_morgan_fingerprint(smiles=smiles)
    input_features = torch.cat([feature_tensor, smiles_feature], dim=-1)
    input_tensor = input_tensor.unsqueeze(0) 
    atoms, bonds,atoms_type=parse_mol2(file)
    graph=create_graph(atoms=atoms,bonds=bonds)
    prediction, _, _ = model(graph, input_tensor, atoms_type)
if __name__ == '__main__':
    prediction()

