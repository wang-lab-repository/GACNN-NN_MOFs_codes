import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import GraphConv,GATConv
import numpy as np
import time
# 图神经网络（GNN）
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph, feat):
        node_features = feat
        edge_features = graph.edata['feat']
        with graph.local_scope():
            graph.ndata['h'] = node_features
            graph.edata['h'] = edge_features
            graph.update_all(message_func=dgl.function.u_add_e('h', 'h', 'm'), reduce_func=dgl.function.sum('m', 'h'))
            h = graph.ndata['h']
        
        # 2. 使用线性层进行处理
        h = F.relu(self.fc1(h))
        h = self.fc2(h)

        return h

class GCNN(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim) -> None:
        super(GCNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, graph, features):
        graph = dgl.add_self_loop(graph)
        # 第一层图卷积
        x = self.conv1(graph, features)
        x = torch.relu(x)
        # 第二层图卷积
        x = self.conv2(graph, x)
        return x
    
class GACNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads) -> None:
        super(GACNN,self).__init__()
        self.layer1 = GATConv(input_dim, hidden_dim, num_heads)
        self.layer2 = GATConv(hidden_dim * num_heads, output_dim, 1)

    def forward(self, graph, features):
        # 第一层 GAT
        graph = dgl.add_self_loop(graph)
        x = self.layer1(graph, features)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)  # 合并多头注意力的输出
        # print(x.shape)
        # 第二层 GAT
        x = self.layer2(graph, x)
        x = torch.squeeze(x, dim=1) 
        return x
class NN(nn.Module):
    def __init__(self, fc_input_dim, fc_hidden_dim, fc_output_dim) -> None:
        super(NN,self).__init__()
        self.fc1 = nn.Linear(fc_input_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_output_dim)
        pass
    def forward(self,x):
        # print("the shape of non_graph is", x.shape)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
class MultimodalAttentionFusionWithResidual(nn.Module):
    def __init__(self, input_dim1, input_dim2, embed_dim=128, num_heads=4):
        """
        带残差连接的多模态多头注意力融合
        
        """
        super().__init__()
        input_dims = [input_dim1,input_dim2]
        self.num_modalities = len(input_dims)
        
        # 特征投影层
        self.projections = nn.ModuleList([
            nn.Linear(dim, embed_dim) for dim in input_dims
        ])
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False
        )
        
        # 残差相关组件
        self.layer_norm_after_attn = nn.LayerNorm(embed_dim)  # 三维输入兼容
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)  # 可选dropout

    def forward(self, modality_features):
        # 1. 特征投影
        projected = [proj(feat) for proj, feat in zip(self.projections, modality_features)]
        
        # 2. 堆叠为序列格式 (num_modalities, batch_size, embed_dim)
        stacked = torch.stack(projected, dim=0)
        
        # 3. 应用多头注意力
        attn_output, _ = self.multihead_attn(
            query=stacked,
            key=stacked,
            value=stacked
        )
        
        # 4. 残差连接 + LayerNorm
        attn_output = self.layer_norm_after_attn(attn_output + stacked)
        
        # 5. 特征聚合（平均池化）
        fused = torch.mean(attn_output, dim=0)
        
        # 6. 最终处理（含残差）
        residual = fused  # 保存原始融合结果
        fused = self.fc(self.dropout(fused))
        fused = self.final_layer_norm(fused + residual)  # 残差连接
        
        return fused   

# # 多模态模型：图卷积网络 + 全连接神经网络
# class MultiModalModel(nn.Module):
#     def __init__(self, graph_input_dim, graph_hidden_dim, graph_output_dim, fc_input_dim, fc_hidden_dim, fc_output_dim):
#         super(MultiModalModel, self).__init__()
        
#         # 图卷积网络部分
#         self.gnn = GCNN(graph_input_dim, graph_hidden_dim, graph_output_dim)
#         self.nn = NN(fc_input_dim, fc_hidden_dim, fc_output_dim)
#         # # 全连接网络部分（用于非图数据：数值特征 + SMILES特征）
#         # self.fc1 = nn.Linear(fc_input_dim, fc_hidden_dim)
#         # self.fc2 = nn.Linear(fc_hidden_dim, fc_output_dim)
        
#         # 最终的特征融合部分
#         self.fc3 = nn.Linear(graph_output_dim + fc_output_dim, 1)  # 预测目标值

#     def forward(self, graph,  non_graph_features,atoms_type, node_indices=None):
        
#         batched_graph = graph # 批处理多个图
#         batched_graph_output = self.gnn(batched_graph, batched_graph.ndata['feat'])
        
#         # 对于每个图，将其节点的特征进行平均（即池化）
#         graph_output = batched_graph_output.split(batched_graph.batch_num_nodes().tolist()) # 按图划分
#         graph_output = [g.mean(dim=0) for g in graph_output]  # 每个图的节点特征取平均
#         graph_output = torch.stack(graph_output)  # 将所有图的输出合并成一个张量
        
#         # 2. 处理非图数据：全连接网络
        
#         non_graph_output = self.nn(non_graph_features)
        
#         # print("the shape of graph is:",graph_output.shape)
#         # print("the shape of non_graph is:", non_graph_output.shape)
#         # 3. 特征融合：将图特征和非图特征拼接
#         combined_features = torch.cat([graph_output, non_graph_output], dim=-1)

#         # 4. 最终输出：预测目标
#         prediction = self.fc3(combined_features)

#         return prediction, atoms_type, node_indices
#     def save_gnn(self):
#         name = time.strftime('checkpoints/'+ '%m_%d_%H-%M-%S_GCNN.pth')
        
#         torch.save(self.gnn.state_dict(), name)
#         pass
#     def save_nn(self):
#         name = time.strftime('checkpoints/'+ '%m_%d_%H-%M-%S_NN.pth')
#         torch.save(self.nn.state_dict(), name)
#         pass
# 多模态模型：图卷积网络 + 全连接神经网络,使用多头注意力机制融合
# class MultiModalModel(nn.Module):
#     def __init__(self, graph_input_dim, graph_hidden_dim, graph_output_dim, fc_input_dim, fc_hidden_dim, fc_output_dim,embed_dim=128, num_heads=4):
#         super(MultiModalModel, self).__init__()
        
#         # 图卷积网络部分
#         self.gnn = GCNN(graph_input_dim, graph_hidden_dim, graph_output_dim)
#         self.nn = NN(fc_input_dim, fc_hidden_dim, fc_output_dim)
#         # # 全连接网络部分（用于非图数据：数值特征 + SMILES特征）
#         # self.fc1 = nn.Linear(fc_input_dim, fc_hidden_dim)
#         # self.fc2 = nn.Linear(fc_hidden_dim, fc_output_dim)
        
#         # 最终的特征融合部分
#         self.mulattfusion = MultimodalAttentionFusionWithResidual(input_dim1=graph_output_dim,input_dim2=fc_output_dim,embed_dim=embed_dim,num_heads=num_heads)
#         self.fc3 = nn.Linear(embed_dim, 1)  # 预测目标值

#     def forward(self, graph,  non_graph_features,atoms_type, node_indices=None):
        
#         batched_graph = graph # 批处理多个图
#         batched_graph_output = self.gnn(batched_graph, batched_graph.ndata['feat'])
        
#         # 对于每个图，将其节点的特征进行平均（即池化）
#         graph_output = batched_graph_output.split(batched_graph.batch_num_nodes().tolist()) # 按图划分
#         graph_output = [g.mean(dim=0) for g in graph_output]  # 每个图的节点特征取平均
#         graph_output = torch.stack(graph_output)  # 将所有图的输出合并成一个张量
        
#         # 2. 处理非图数据：全连接网络
        
#         non_graph_output = self.nn(non_graph_features)
        
#         # print("the shape of graph is:",graph_output.shape)
#         # print("the shape of non_graph is:", non_graph_output.shape)
#         # 3. 特征融合：将图特征和非图特征拼接
#         combined_features = self.mulattfusion([graph_output,non_graph_output])

#         # 4. 最终输出：预测目标
#         prediction = self.fc3(combined_features)

#         return prediction, atoms_type, node_indices
#     def save_gnn(self):
#         name = time.strftime('checkpoints/'+ '%m_%d_%H-%M-%S_GCNN.pth')
        
#         torch.save(self.gnn.state_dict(), name)
#         pass
#     def save_nn(self):
#         name = time.strftime('checkpoints/'+ '%m_%d_%H-%M-%S_NN.pth')
#         torch.save(self.nn.state_dict(), name)
#         pass
# 多模态模型：图注意力网络 + 全连接神经网络,使用多头注意力机制融合
class MultiModalModel(nn.Module):
    def __init__(self, graph_input_dim, graph_hidden_dim, graph_output_dim, fc_input_dim, fc_hidden_dim, fc_output_dim,embed_dim=128, num_heads=4):
        super(MultiModalModel, self).__init__()
        
        # 图注意力网络部分
        # self.gnn = GCNN(graph_input_dim, graph_hidden_dim, graph_output_dim)
        self.gnn = GACNN(graph_input_dim, graph_hidden_dim, graph_output_dim,num_heads)
        self.nn = NN(fc_input_dim, fc_hidden_dim, fc_output_dim)
        # # 全连接网络部分（用于非图数据：数值特征 + SMILES特征）
        # self.fc1 = nn.Linear(fc_input_dim, fc_hidden_dim)
        # self.fc2 = nn.Linear(fc_hidden_dim, fc_output_dim)
        
        # 最终的特征融合部分
        self.mulattfusion = MultimodalAttentionFusionWithResidual(input_dim1=graph_output_dim,input_dim2=fc_output_dim,embed_dim=embed_dim,num_heads=num_heads)
        self.fc3 = nn.Linear(embed_dim, 16)  # 预测目标值
        self.fc4 = nn.Linear(16,1)
        self.drop = nn.Dropout(p=0.05)
        self.relu = nn.ReLU()
    def forward(self, graph,  non_graph_features,atoms_type, node_indices=None):
        
        batched_graph = graph # 批处理多个图
        batched_graph_output = self.gnn(batched_graph, batched_graph.ndata['feat'])
        
        # 对于每个图，将其节点的特征进行平均（即池化）
        graph_output = batched_graph_output.split(batched_graph.batch_num_nodes().tolist()) # 按图划分
        graph_output = [g.mean(dim=0) for g in graph_output]  # 每个图的节点特征取平均
        graph_output = torch.stack(graph_output)  # 将所有图的输出合并成一个张量
        
        # 2. 处理非图数据：全连接网络
        
        non_graph_output = self.nn(non_graph_features)
        
        # print("the shape of graph is:",graph_output.shape)
        # print("the shape of non_graph is:", non_graph_output.shape)
        # 3. 特征融合：将图特征和非图特征拼接
        combined_features = self.mulattfusion([graph_output,non_graph_output])

        # 4. 最终输出：预测目标
        x = self.fc3(combined_features)
        x = self.drop(x)
        x = self.relu(x)
        prediction = self.fc4(x)
        # prediction = self.fc3(combined_features)
        return prediction, atoms_type, node_indices
    def save_gnn(self):
        name = time.strftime('checkpoints/'+ '%m_%d_%H-%M-%S_GACNN.pth')
        
        torch.save(self.gnn.state_dict(), name)
        pass
    def save_nn(self):
        name = time.strftime('checkpoints/'+ '%m_%d_%H-%M-%S_NN.pth')
        torch.save(self.nn.state_dict(), name)
        pass

# # 多模态模型：图神经网络 + 全连接神经网络,使用多头注意力机制融合
# class MultiModalModel(nn.Module):
#     def __init__(self, graph_input_dim, graph_hidden_dim, graph_output_dim, fc_input_dim, fc_hidden_dim, fc_output_dim,embed_dim=128, num_heads=4):
#         super(MultiModalModel, self).__init__()
        
#         # 网络部分
        
#         self.gnn = GNN(graph_input_dim, graph_hidden_dim, graph_output_dim)
#         self.nn = NN(fc_input_dim, fc_hidden_dim, fc_output_dim)
        
        
#         # 最终的特征融合部分
#         self.mulattfusion = MultimodalAttentionFusionWithResidual(input_dim1=graph_output_dim,input_dim2=fc_output_dim,embed_dim=embed_dim,num_heads=num_heads)
#         self.fc3 = nn.Linear(embed_dim, 1)  # 预测目标值
#         self.weight = nn.Parameter(torch.randn(1))
#         self.fc3 = nn.Linear(embed_dim, 16)  # 预测目标值
#         self.fc4 = nn.Linear(16,1)
#         self.drop = nn.Dropout(p=0.05)
#         self.relu = nn.ReLU()
#     def forward(self, graph,  non_graph_features,atoms_type, node_indices=None):
        
#         batched_graph = graph # 批处理多个图
#         batched_graph_output = self.gnn(batched_graph, batched_graph.ndata['feat'])
        
#         # 对于每个图，将其节点的特征进行平均（即池化）
#         graph_output = batched_graph_output.split(batched_graph.batch_num_nodes().tolist()) # 按图划分
#         graph_output = [g.mean(dim=0) for g in graph_output]  # 每个图的节点特征取平均
#         graph_output = torch.stack(graph_output)  # 将所有图的输出合并成一个张量
        
#         # 2. 处理非图数据：全连接网络
        
#         non_graph_output = self.nn(non_graph_features)
        
#         # print("the shape of graph is:",graph_output.shape)
#         # print("the shape of non_graph is:", non_graph_output.shape)
#         # 3. 特征融合：将图特征和非图特征拼接
#         combined_features = self.mulattfusion([graph_output,non_graph_output])
#         # combined_features = torch.cat([graph_output, non_graph_output], dim=-1)
#         # combined_features = graph_output + non_graph_output
#         # combined_features = self.weight * graph_output + (1-self.weight) * non_graph_output
#         # 4. 最终输出：预测目标
#         x = self.fc3(combined_features)
#         x = self.drop(x)
#         x = self.relu(x)
#         prediction = self.fc4(x)

#         return prediction, atoms_type, node_indices
#     def save_gnn(self):
#         name = time.strftime('checkpoints/'+ '%m_%d_%H-%M-%S_GNN.pth')
        
#         torch.save(self.gnn.state_dict(), name)
#         pass
#     def save_nn(self):
#         name = time.strftime('checkpoints/'+ '%m_%d_%H-%M-%S_NN.pth')
#         torch.save(self.nn.state_dict(), name)
#         pass

if __name__ == '__main__':
    from config import Config
    from data import MultiModalDataset
    opt = Config()
    dataset = MultiModalDataset(excel_path=opt.excel_path,graph_file_dir=opt.graph_file_dir)
    graph = dataset[0][0]
    # model = GCNN(input_dim=opt.graph_input_dim,hidden_dim=opt.graph_hidden_dim,output_dim=opt.graph_output_dim)
    model = GACNN(input_dim=opt.graph_input_dim,hidden_dim=opt.graph_hidden_dim,output_dim=opt.graph_output_dim,num_heads=opt.graph_num_heads)
    output = model(graph,graph.ndata['feat'])
    print(output.shape)
    