import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,mean_absolute_percentage_error
from joblib import dump, load
from rdkit import Chem
from rdkit.Chem import AllChem
from config import Config
from tool import smiles_to_morgan_fingerprint, smiles_to_atom_pair_fingerprint, smiles_to_maccs_fingerprint,smiles_to_rdkit_fingerprint,smiles_to_topological_fingerprint
# 自定义函数：将SMILES转换为Morgan Fingerprint


class MyData:
    def __init__(self, df):
        self.df = df
        
        self.features_columns = ['Al', 'Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Ni', 'Zn', 'Zr', 
                                 'initial concentration of adsorbates(ppm)', 'initial concentration of MOF(mg/ml)', 
                                 'Metal loading(wt.%)', 'Sonochemical', 'Microwave', 'mechanochemical', 'hydrothermal', 
                                 'Solvothermal', 'Monoclinic', 'Tetragonal', 'orthorhombic', 'trigonal', 'cubic', 'Triclinic', 'hexagonal', 'time(min)']
        self.target_column = 'Adsorption(mg/g)'
        self.smiles_column = 'Adsorbates'
        
    def get_data(self):
        features = []
        targets = []
        
        for _, row in self.df.iterrows():
            # 获取数值特征
            feature_values = row[self.features_columns].values.astype(np.float32)
            target_value = row[self.target_column]
            
            # 获取SMILES特征
            smiles = row[self.smiles_column]
            smiles_features = smiles_to_morgan_fingerprint(smiles)
            # smiles_features = smiles_to_topological_fingerprint(smiles=smiles)
            # smiles_features = smiles_to_atom_pair_fingerprint(smiles=smiles)
            # smiles_features = smiles_to_maccs_fingerprint(smiles=smiles)
            # smiles_features = smiles_to_rdkit_fingerprint(smiles=smiles)
            # 拼接特征
            all_features = np.concatenate([feature_values, smiles_features])
            
            features.append(all_features)
            targets.append(target_value)
        
        return np.array(features), np.array(targets)

# 加载数据
opt = Config()


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
train_remaining, test_remaining = train_test_split(df_remaining, test_size=0.2, random_state=160, stratify=df_remaining['combination'])

# 合并两个数据集
train_df = pd.concat([train_remaining, train_single])
test_df = pd.concat([test_remaining, test_single])
# train_indices, test_indices = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['combination'])
# train_df = df.loc[train_indices]
# test_df = df.loc[test_indices]
# print(len(train_df))
# print(len(test_df))
train_dataset = MyData(train_df)
test_dataset = MyData(test_df)
# 获取特征和目标
# X, y = dataset.get_data()
# # X_combined = 
# # 拆分训练集和测试集
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48,stratify=X_combined)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, y_train = train_dataset.get_data()
X_test, y_test = test_dataset.get_data()
# 使用 RandomForestRegressor 进行训练
rf = RandomForestRegressor(n_estimators=100, random_state=80)
rf.fit(X_train, y_train)
dump(rf, 'checkpoints/random_forest_model.joblib')
train_pre = rf.predict(X_train)
test_pre = rf.predict(X_test)
train_r2 = r2_score(y_train,train_pre)
test_r2 = r2_score(y_test,test_pre)
train_mape = mean_absolute_percentage_error(y_train,train_pre)
test_mape = mean_absolute_percentage_error(y_test,test_pre)
train_rmse = np.sqrt(((train_pre - y_train) ** 2).mean())
test_rmse = np.sqrt(((test_pre - y_test) ** 2).mean())
print("----the result of the rf on train -----")
print("R2 is:", train_r2)
print("RMSE is:",train_rmse)
print("MAPE is:",train_mape)
print("----the result of the rf on test -----")
print("R2 is:", test_r2)
print("RMSE is:",test_rmse)
print("MAPE is:",test_mape)

gbr = GradientBoostingRegressor(n_estimators=180,random_state=120)
gbr.fit(X_train, y_train)

train_pre = gbr.predict(X_train)
test_pre = gbr.predict(X_test)
train_r2 = r2_score(y_train,train_pre)
test_r2 = r2_score(y_test,test_pre)
train_mape = mean_absolute_percentage_error(y_train,train_pre)
test_mape = mean_absolute_percentage_error(y_test,test_pre)
train_rmse = np.sqrt(((train_pre - y_train) ** 2).mean())
test_rmse = np.sqrt(((test_pre - y_test) ** 2).mean())
print("----the result of the gbr on train -----")
print("R2 is:", train_r2)
print("RMSE is:",train_rmse)
print("MAPE is:",train_mape)
print("----the result of the gbr on test -----")
print("R2 is:", test_r2)
print("RMSE is:",test_rmse)
print("MAPE is:",test_mape)

ada = AdaBoostRegressor(n_estimators=180,random_state=80)
ada.fit(X_train, y_train)

train_pre = ada.predict(X_train)
test_pre = ada.predict(X_test)
train_r2 = r2_score(y_train,train_pre)
test_r2 = r2_score(y_test,test_pre)
train_mape = mean_absolute_percentage_error(y_train,train_pre)
test_mape = mean_absolute_percentage_error(y_test,test_pre)
train_rmse = np.sqrt(((train_pre - y_train) ** 2).mean())
test_rmse = np.sqrt(((test_pre - y_test) ** 2).mean())
print("----the result of the adaboost on train -----")
print("R2 is:", train_r2)
print("RMSE is:",train_rmse)
print("MAPE is:",train_mape)
print("----the result of the adaboost on test -----")
print("R2 is:", test_r2)
print("RMSE is:",test_rmse)
print("MAPE is:",test_mape)

mlp = MLPRegressor(hidden_layer_sizes=(256,64,8))
mlp.fit(X_train, y_train)

train_pre = mlp.predict(X_train)
test_pre = mlp.predict(X_test)
train_r2 = r2_score(y_train,train_pre)
test_r2 = r2_score(y_test,test_pre)
train_mape = mean_absolute_percentage_error(y_train,train_pre)
test_mape = mean_absolute_percentage_error(y_test,test_pre)
train_rmse = np.sqrt(((train_pre - y_train) ** 2).mean())
test_rmse = np.sqrt(((test_pre - y_test) ** 2).mean())
print("----the result of the mlp on train -----")
print("R2 is:", train_r2)
print("RMSE is:",train_rmse)
print("MAPE is:",train_mape)
print("----the result of the mlp on test -----")
print("R2 is:", test_r2)
print("RMSE is:",test_rmse)
print("MAPE is:",test_mape)