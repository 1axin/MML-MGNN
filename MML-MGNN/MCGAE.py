# 作者:     wxf

# 开发时间: 2024/5/22 17:01
import csv
import argparse
import csv
import random

from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from catboost import CatBoostClassifier
from numpy import *
from torch_geometric.nn import GAE
import torch
import torch.nn as nn
from numpy import interp
from torch_geometric.nn import GCNConv, ChebConv, DNAConv
from torch_geometric.nn import GAE , ARGA
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges, negative_sampling, degree
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import random
import string
from sklearn import metrics
from torch_geometric.data import Data, download_url, extract_gz
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GAEncoder(nn.Module) :

    def __init__(self, in_channels, hidden_size, out_channels, dropout) :
        super(GAEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv2 = GCNConv(hidden_size, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X1, X2, X3, edge_index) :
        # first channel
        x1 = self.conv1(X1, edge_index).relu()
        x1 = F.dropout(x1, training=self.training)
        x1 = self.conv2(x1, edge_index)
        # second channel
        x2 = self.conv1(X2, edge_index).relu()
        x2 = F.dropout(x2, training=self.training)
        x2 = self.conv2(x2, edge_index)
        # th channel
        x3 = self.conv1(X3, edge_index).relu()
        x3 = F.dropout(x3, training=self.training)
        x3 = self.conv2(x3, edge_index)

        out = (x1 + x2 + x3) / 3



        return out


def train(train_data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x1, train_data.x2, train_data.x3, train_data.edge_index)

    loss = model.recon_loss(z, train_data.pos_edge_label_index.to(device))
    loss.backward(retain_graph=True)
    optimizer.step()
    return float(loss), z

@torch.no_grad()
def gae_test(test_data,model):
    model.eval()
    z = model.encode(test_data.x1, test_data.x2, test_data.x3,test_data.edge_index)
    loss = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
    return loss


def MyData():
    # 数据构建

    # 读取CSV文件
    data = pd.read_csv('MDA预测_全部正样本.csv', header = None)

    # 提取所有节点并按升序排列
    nodes = sorted(list(set(data[0].unique()) | set(data[1].unique())))

    # 构建节点字典，将节点映射为索引
    node_index = {node : idx for idx, node in enumerate(nodes)}

    # 构建边索引
    edge_index = []
    for row in data.iterrows() :
        node1 = row[1][0]
        node2 = row[1][1]
        edge_index.append((node_index[node1], node_index[node2]))
        # edge_index.append((node_index[node2], node_index[node1]))  # 无向图，需要考虑反向边
    for row in data.iterrows() :
        node1 = row[1][0]
        node2 = row[1][1]
        edge_index.append((node_index[node2], node_index[node1]))  # 无向图，需要考虑反向边
    # 输出边索引
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_index = torch.tensor(edge_index, dtype = torch.long)
    print(edge_index)
    print(edge_index.shape)

    # 获取第一列数据并保持原始顺序
    first_column = edge_index[0].tolist()

    # 去重复并保持原始顺序
    unique_edges = list(dict.fromkeys(first_column))

    # 转换为 DataFrame，然后保存为 CSV 文件
    df = pd.DataFrame(unique_edges)
    df.to_csv('unique_nodes.csv', index = False, header = None)

    # 读入节点模态1特征 CSV 文件
    node_features_df1 = pd.read_csv('All_Feature_全连接降维.csv', header = None)

    df_as_list = df.values.tolist()
    node_features_df_as_list1 = node_features_df1.values.tolist()

    SampleFeature = []
    counter = 0
    while counter < len(df_as_list) :
        counter1 = 0
        while counter1 < len(node_features_df_as_list1) :
            if df_as_list[counter][0] == node_features_df_as_list1[counter1][0] :
                a = []
                a.extend(node_features_df_as_list1[counter1][1 :])
                break
            else :
                a = [0] * (64)
            counter1 = counter1 + 1
        SampleFeature.append(a)
        counter = counter + 1
    storFile(SampleFeature, 'unique_nodes_Feature_X1.csv')

    result1 = pd.DataFrame(SampleFeature)

    # # 每个节点的特征
    X1 = result1.values
    X1 = torch.tensor(X1, dtype = torch.float)
    print(X1)
    print(X1.shape)

    # 读入节点模态2特征 CSV 文件
    node_features_df2 = pd.read_csv('All_Feature_DW_全连接降维.csv', header = None)

    node_features_df_as_list2 = node_features_df2.values.tolist()

    SampleFeature = []
    counter = 0
    while counter < len(df_as_list) :
        counter1 = 0
        while counter1 < len(node_features_df_as_list2) :
            if df_as_list[counter][0] == node_features_df_as_list2[counter1][0] :
                a = []
                a.extend(node_features_df_as_list2[counter1][1 :])
                break
            else :
                a = [0] * (64)
            counter1 = counter1 + 1
        SampleFeature.append(a)
        counter = counter + 1
    storFile(SampleFeature, 'unique_nodes_Feature_X2.csv')

    result2 = pd.DataFrame(SampleFeature)

    # # 每个节点的特征
    X2 = result2.values
    X2 = torch.tensor(X2, dtype = torch.float)
    print(X2)
    print(X2.shape)

    # 读入节点模态3特征 CSV 文件
    node_features_df3 = pd.read_csv('All_Feature_ST_全连接降维.csv', header = None)

    node_features_df_as_list3 = node_features_df3.values.tolist()

    SampleFeature = []
    counter = 0
    while counter < len(df_as_list) :
        counter1 = 0
        while counter1 < len(node_features_df_as_list3) :
            if df_as_list[counter][0] == node_features_df_as_list3[counter1][0] :
                a = []
                a.extend(node_features_df_as_list3[counter1][1 :])
                break
            else :
                a = [0] * (64)
            counter1 = counter1 + 1
        SampleFeature.append(a)
        counter = counter + 1
    storFile(SampleFeature, 'unique_nodes_Feature_X3.csv')

    result3 = pd.DataFrame(SampleFeature)

    # # 每个节点的特征
    X3 = result3.values
    X3 = torch.tensor(X3, dtype = torch.float)
    print(X3)
    print(X3.shape)


    data = Data(x1 = X1, x2 = X2, x3 = X3, edge_index = edge_index)

    print('数据的键', data.keys)
    # 打印图所有键
    # 输出：['edge_index', 'x']
    print('分子特征', data['x1'], data['x2'], data['x3'])
    # 打印节点特征向量
    # 输出：
    # tensor([[-1.],
    #        [ 0.],
    #        [ 1.]])
    print('边', data['edge_index'])
    # 打印边的稀疏矩阵
    # 输出：
    # tensor([[0, 1, 1, 2],
    #        [1, 0, 2, 1]])
    print('节点数量', data.num_nodes)
    # 打印节点个数
    # 输出：3
    print('边的数量', data.num_edges)
    # 打印边的条数
    # 输出：4
    print('节点特征维度', data.num_node_features)
    # 打印节点特征维度
    # 输出：1
    print('是否有孤立节点', data.has_isolated_nodes())
    # 判断是否有孤立点
    # 输出：FALSE
    print('是否有自循环', data.has_self_loops())
    # 判断是否有自循环
    # 输出：FALSE
    print('是否为无向图', data.is_undirected())
    # 判断是否为无向图
    # 输出：True
    return data

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

data = MyData()

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.2, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=True),
])
train_datasets, val_datasets, test_datasets = transform(data)
print("Train Data:", train_datasets)
print("Validation Data:", val_datasets)
print("Test Data:", test_datasets)


model = GAE(GAEncoder(in_channels = train_datasets.x1.shape[1], hidden_size = 64, out_channels = 32, dropout = 0.2
             )).to(device)
print(model)


optimizer = optim.Adam(model.parameters(),lr=0.001)
losses = []
test_auc = []
test_ap = []
train_aucs = []
train_aps = []
train_loss=[]
test_acc=[]
save_interval = 20  # 设定保存轮数间隔为50
for epoch in range(1, 20):

    # losses = []
    # test_auc = []
    # test_ap = []
    # train_aucs = []
    # train_aps = []

    loss, z = train(train_datasets, model, optimizer)

    if (epoch + 1) % save_interval == 0 :  # 每50轮保存一次节点特征为 CSV 文件
        # 将节点特征 out 转换为 DataFrame
        out_fea = pd.DataFrame(z.detach().numpy())  # 假设 out 是 PyTorch Tensor 类型，需要通过 .detach().numpy() 方法转换为 NumPy 数组
        # 保存 DataFrame 到 CSV 文件
        # 所有节点的GCN输出特征保存
        out_fea.to_csv(f'GAE_node_features_epoch_{epoch + 1}.csv', header = None, index = False)  # 保存为名为 node_features_epoch_{epoch}.csv 的文件，不保存行索引

    losses.append(loss)
    auc1, ap = gae_test(test_datasets, model)
    test_auc.append(auc1)
    test_ap.append(ap)

    train_auc, train_ap = gae_test(train_datasets, model)

    train_aucs.append(train_auc)
    train_aps.append(train_ap)
    train_loss.append(loss)
    # test_acc.append(tst_acc)
    print('Epoch: {:03d}, test AUC: {:.4f}, test AP: {:.4f}, train AUC: {:.4f}, train AP: {:.4f}, loss:{:.4f}'.format(epoch, auc1, ap, train_auc, train_ap, loss))


# 绘制训练损失和验证损失的折线图
epochs = range(1, len(train_loss) + 1)  # 横轴表示 epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.title(f'Train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"pictures_loss_{epoch+1}.tif")
plt.show()

# 分类器

# 构建训练样本
# 训练样本关联
train_sample_all = torch.cat(( train_datasets.pos_edge_label_index, train_datasets.neg_edge_label_index), dim = 1)
# train_sample_all = pd.DataFrame(train_sample_all.detach().numpy())

train_sample = pd.DataFrame(train_sample_all.detach().numpy().T, columns = None)
train_sample.to_csv('train_sample.csv', header = None, index = False)
# 验证样本的标签
train_label_all = torch.cat(( train_datasets.pos_edge_label, train_datasets.neg_edge_label), dim = 0)
# train_label_all = pd.DataFrame(train_datasets.detach().numpy())

train_label = pd.DataFrame(train_label_all.detach().numpy(), columns = None)
train_label.to_csv('train_label.csv', header = None, index = False)


# 构建验证样本
# 验证样本关联
val_sample_all = torch.cat(( val_datasets.pos_edge_label_index, val_datasets.neg_edge_label_index), dim = 1)
# val_sample_all = pd.DataFrame(val_sample_all.detach().numpy())

val_sample = pd.DataFrame(val_sample_all.detach().numpy().T, columns = None)
val_sample.to_csv('val_sample.csv', header = None, index = False)
# 验证样本的标签
val_label_all = torch.cat(( val_datasets.pos_edge_label, val_datasets.neg_edge_label), dim = 0)
# val_label_all = pd.DataFrame(val_label_all.detach().numpy())

# val_label = pd.DataFrame(val_label_all.detach().numpy(), columns = None)
# val_label.to_csv('val_label.csv', header = None, index = False)


# 手打的MDA标签
data_label = []
data1 = []
data1 = ones((1,782), dtype=int)
# 2532 1933
data2 = zeros((1, 782))
data_label.extend(data1[0])
data_label.extend(data2[0])
SampleLabel = data_label

# 读取GCN生成的每个结点的特征 并加上结点的序号
# 匹配验证样本的特征 用于分类器预测
GCN_ALL_Feature = pd.read_csv('GAE_node_features_epoch_20.csv', header = None)
ALL_Node = pd.read_csv('unique_nodes.csv', header = None)
ALL_Node_GCN_Feature = pd.concat([ALL_Node, GCN_ALL_Feature], axis=1)
ALL_Node_GCN_Feature.to_csv('ALL_Node_GCN_Feature.csv', header = None)

# 训练样本特征匹配

# 读取保存的文件并进行匹配
data = []
ReadMyCsv(data, "train_sample.csv")
print(np.array(data).shape)

AllSample = data

Allnode = []
ReadMyCsv(Allnode, "ALL_Node_GCN_Feature.csv")
print(np.array(Allnode).shape)

# SampleFeature
SampleFeature_train = []
counter = 0
while counter < len(AllSample) :
    counter1 = 0
    while counter1 < len(Allnode) :
        if AllSample[counter][0] == Allnode[counter1][0] :
            a = []
            a.extend(Allnode[counter1][2 :])
            break
        counter1 = counter1 + 1

    counter2 = 0
    while counter2 < len(Allnode) :
        if AllSample[counter][1] == Allnode[counter2][0] :
            b = []
            b.extend(Allnode[counter2][2 :])
        counter2 = counter2 + 1
    a.extend(b)
    SampleFeature_train.append(a)
    counter = counter + 1
counter1 = 0
storFile(SampleFeature_train, 'train_data_SampleFeature_分类器.csv')
print('SampleFeature_train', np.array(SampleFeature_train).shape)

# 测试样本特征匹配

# 读取保存的文件并进行匹配
data = []
ReadMyCsv(data, "MDA预测_测试集正负样本.csv")
print(np.array(data).shape)

AllSample = data

Allnode = []
ReadMyCsv(Allnode, "ALL_Node_GCN_Feature.csv")
print(np.array(Allnode).shape)

# SampleFeature
SampleFeature_val = []
counter = 0
while counter < len(AllSample) :
    counter1 = 0
    while counter1 < len(Allnode) :
        if AllSample[counter][0] == Allnode[counter1][0] :
            a = []
            a.extend(Allnode[counter1][2 :])
            break
        counter1 = counter1 + 1

    counter2 = 0
    while counter2 < len(Allnode) :
        if AllSample[counter][1] == Allnode[counter2][0] :
            b = []
            b.extend(Allnode[counter2][2 :])
        counter2 = counter2 + 1
    a.extend(b)
    SampleFeature_val.append(a)
    counter = counter + 1
counter1 = 0
storFile(SampleFeature_val, 'val_data_SampleFeature_分类器.csv')
print('SampleFeature_val', np.array(SampleFeature_val).shape)

SampleFeature_train_feature = np.array(SampleFeature_train)
SampleFeature_train_label = np.array(train_label.values.tolist())
SampleFeature_val_feature = np.array(SampleFeature_val)
SampleFeature_val_label = np.array(data_label)

# 随机打乱

permutation = np.random.permutation(SampleFeature_train_label.shape[0])
SampleFeature_train_feature = SampleFeature_train_feature[permutation, :]
SampleFeature_train_label = SampleFeature_train_label[permutation]

permutation = np.random.permutation(SampleFeature_val_label.shape[0])
SampleFeature_val_feature = SampleFeature_val_feature[permutation, :]
SampleFeature_val_label = SampleFeature_val_label[permutation]

# 引入分类器进行训练 预测
mean_average_precision = []
mean_precision = 0.0
Y_test = []
Y_pre = []
Y_all = []
result = []
mean_recall = np.linspace(0, 1, 10000)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 10000)
for i in range(5):

    print("==================", i, "fold", "==================")
    model_Classifier = CatBoostClassifier(iterations=300, learning_rate=0.006,)
    # model_Classifier = GradientBoostingClassifier(n_estimators=100, max_depth=5,)
    predicted = model_Classifier.fit(SampleFeature_train_feature, SampleFeature_train_label).predict_proba(SampleFeature_val_feature)
    fpr, tpr, thresholds = roc_curve(SampleFeature_val_label, predicted[:, 1])

    precision, recall, _ = precision_recall_curve(SampleFeature_val_label, predicted[:, 1])
    average_precision = average_precision_score(SampleFeature_val_label, predicted[:, 1])
    mean_average_precision.append(average_precision)
    mean_precision += interp(mean_recall, recall, precision)

    # 假设 threshold 是你设定的阈值
    threshold = 0.5
    # 将连续预测值转换为离散类型
    discrete_predictions = np.where(predicted[:, 1] > threshold, 1, 0)
    # 计算准确率
    acc = accuracy_score(SampleFeature_val_label, discrete_predictions)
    print("Test accuracy: ", acc)
    # 输出分类报告
    print("Classification Report:")
    print(classification_report(SampleFeature_val_label, discrete_predictions, digits=4))
    # 输出混淆矩阵
    print("Confusion Matrix:")
    print(confusion_matrix(SampleFeature_val_label, discrete_predictions))

    result.append(classification_report(SampleFeature_val_label, discrete_predictions, digits=4))
    with open('classification_report.txt', 'w') as f :
        f.write(classification_report(SampleFeature_val_label, discrete_predictions, digits=4))

    Y_test1 = []
    Y_pre1 = []

    Y_test1.extend(SampleFeature_val_label)
    Y_pre1.extend(predicted[:, 1])

    np.save('Y_test'+str(i),Y_test1)
    np.save('Y_pre' + str(i), Y_pre1)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw = 1.5, alpha = 1,
             label = 'ROC fold %d (AUC = %0.4f)' % (i , roc_auc))

conf_matrix_df = pd.DataFrame(result).transpose()
conf_matrix_df.to_csv("result.csv", index = False)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=1.5, alpha=1)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc = "lower right")
plt.savefig('ROC-5fold-test.tif')
plt.show()




