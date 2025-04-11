import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dgl
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
from captum.attr import IntegratedGradients
from tool import collate_fn,model_predict,draw_loss,get_R2_RMSE,visualize_graph_bymodel
from data import MultiModalDataset
from model import MultiModalModel
from config import Config
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for graph, input_tensor, target_tensor, atoms_type ,   mof_indexs, smiles in train_loader:
        # 将数据移动到设备上
        graph = graph.to(device)
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        # 通过模型进行预测
        prediction,_,_ = model(graph,  input_tensor,atoms_type)  # 假设graph和non-graph features相同，实际可以不同

        # 计算损失
        loss = criterion(prediction.view(-1), target_tensor)
        total_loss += loss.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

# 定义模型训练所需的元素
def train_model(dataset, graph_input_dim, graph_hidden_dim, graph_output_dim, fc_input_dim, fc_hidden_dim, fc_output_dim,embed_dim, num_heads, epochs=10, batch_size=32, learning_rate=0.001, device='cuda'):
    # 创建 DataLoader
    train_loader = dgl.dataloading.GraphDataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = MultiModalModel(graph_input_dim, graph_hidden_dim, graph_output_dim, fc_input_dim, fc_hidden_dim, fc_output_dim,embed_dim=embed_dim,num_heads=num_heads).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 假设这是回归问题
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    # 训练模型
    for epoch in range(epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        loss_list.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    return model,loss_list

if __name__ == '__main__':
    opt = Config()
    # dataset = MultiModalDataset(excel_path=opt.excel_path,graph_file_dir=opt.graph_file_dir)
    # cdcc_list = pd.read_excel(opt.excel_path).iloc[:,0]
    # # print(cdcc_list)
    # # 获取数据集中的索引
    # indices = list(range(len(dataset)))

    # # 划分训练集和测试集，按 8:2 比例
    # train_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=opt.random_state, stratify=cdcc_list)
    
    # # train_indices, test_indices = train_test_split(indices, test_size=0.15,  stratify=cdcc_list)
    # # 创建训练集和测试集的子集
    # train_subset = torch.utils.data.Subset(dataset, train_indices)
    # test_subset = torch.utils.data.Subset(dataset, test_indices)
    df = pd.read_excel(opt.excel_path)
    df['combination'] = df['mof_id'].astype(str) + "_" + df['Adsorbates']
    # 计算每个 combination 的出现频次
    combination_counts = df['combination'].value_counts()

    # 将只出现一次的 combination 单独提取
    single_occurrence_combinations = combination_counts[combination_counts == 1].index
    df_single = df[df['combination'].isin(single_occurrence_combinations)]

    # 将剩下的 combination 提取出来
    df_remaining = df[~df['combination'].isin(single_occurrence_combinations)]

    # 对于只出现一次的组合，随机分配到训练集和测试集中
    train_single, test_single = train_test_split(df_single, test_size=0.2, random_state=42)

    # 使用 train_test_split 对其余组合进行分层划分
    train_remaining, test_remaining = train_test_split(df_remaining, test_size=0.2, random_state=110, stratify=df_remaining['combination'])

    # 重置索引
    df_single = df_single.reset_index(drop=True)
    df_remaining = df_remaining.reset_index(drop=True)

    # 合并数据
    train_df = pd.concat([train_remaining, train_single])
    test_df = pd.concat([test_remaining, test_single])

    # 再次重置索引，避免合并后的索引问题
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_subset = MultiModalDataset(df=train_df,graph_file_dir=opt.graph_file_dir)
    test_subset = MultiModalDataset(df=test_df,graph_file_dir=opt.graph_file_dir)
    print("the size of train is:",len(train_subset))
    print("the siae of test is:", len(test_subset))
    trained_model, losslist = train_model(dataset=train_subset,
                                graph_input_dim=opt.graph_input_dim,
                                graph_hidden_dim=opt.graph_hidden_dim,
                                graph_output_dim=opt.graph_output_dim,
                                fc_input_dim=opt.fc_input_dim,
                                fc_hidden_dim=opt.fc_hidden_dim,
                                fc_output_dim=opt.fc_output_dim,
                                embed_dim=opt.embed_dim,
                                num_heads=opt.num_heads,
                                epochs=opt.epochs,
                                batch_size=opt.batch_size,
                                learning_rate=opt.lr,
                                device=opt.device
                                )
    torch.save(trained_model.state_dict(), 'checkpoints/GACNN_NN_mulattfusion.pth' )
    trained_model.save_gnn()
    trained_model.save_nn()
    draw_loss(loss_list=losslist)
    train_r2,train_rmse, train_mape = get_R2_RMSE(model=trained_model,dataloader=dgl.dataloading.GraphDataLoader(train_subset, collate_fn=collate_fn, batch_size=opt.batch_size, shuffle=True),device=opt.device)
    print("----the result of the model on train -----")
    print("R2 is:", train_r2)
    print("RMSE is:",train_rmse)
    print("MAPE is:",train_mape)
    test_r2,test_rmse, test_mape = get_R2_RMSE(model=trained_model,dataloader=dgl.dataloading.GraphDataLoader(test_subset, collate_fn=collate_fn, batch_size=opt.batch_size, shuffle=True),device=opt.device)
    print("----the result of the model on test -----")
    print("R2 is:", test_r2)
    print("RMSE is:",test_rmse)
    print("MAPE is:",test_mape)

    
    

    