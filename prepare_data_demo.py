

import csv
import os
import torch as t
import numpy as np
from math import e
import pandas as pd
from scipy import io
from SigmoidKernel import SigmoidKernelDisease
from SigmoidKernel import SigmoidKernelRNA
from hypergraph_construct_KNN import Eu_dis


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def read_mat(path, name):
    matrix = io.loadmat(path)
    matrix = t.FloatTensor(matrix[name])
    return matrix


def read_md_data(path, validation):
    result = [{} for _ in range(validation)]
    for filename in os.listdir(path):
        data_type = filename[filename.index('_')+1:filename.index('.')-1]
        num = int(filename[filename.index('.')-1])
        result[num-1][data_type] = read_csv(os.path.join(path, filename))
    return result


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def Gauss_M(adj_matrix, N):
    GM = np.zeros((N, N))
    rm = N * 1. / sum(sum(adj_matrix * adj_matrix))
    for i in range(N):
        for j in range(N):
            GM[i][j] = e ** (-rm * (np.dot(adj_matrix[i, :] - adj_matrix[j, :], adj_matrix[i, :] - adj_matrix[j, :])))
    return GM


def Gauss_D(adj_matrix, M):
    GD = np.zeros((M, M))
    T = adj_matrix.transpose()
    rd = M * 1. / sum(sum(T * T))
    for i in range(M):
        for j in range(M):
            GD[i][j] = e ** (-rd * (np.dot(T[i] - T[j], T[i] - T[j])))
    return GD

# 定义函数 读取保存文件
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

def association_to_matrix(fileName):
    # 读取 CSV 文件
    df = pd.read_csv(fileName, header=None, names=['node1', 'node2'])

    # 获取所有节点并按升序排列
    nodes_col = sorted(list(set(df['node1'])))  # 第一列节点按升序排列
    nodes_row = sorted(list(set(df['node2'])))  # 第二列节点按升序排列

    # 创建一个空的节点关联矩阵
    num_rows = len(nodes_row)
    num_cols = len(nodes_col)
    adj_matrix = [[0] * num_cols for _ in range(num_rows)]

    # 根据交互对更新节点关联矩阵
    for _, row in df.iterrows():
        node1_idx = nodes_col.index(row['node1'])
        node2_idx = nodes_row.index(row['node2'])
        adj_matrix[node2_idx][node1_idx] = 1

    # 打印节点关联矩阵
    for row in adj_matrix:
        print(row)

    # 将节点关联矩阵保存到 CSV 文件
    adj_df = pd.DataFrame(adj_matrix, columns=nodes_col, index=nodes_row)
    # adj_df.to_csv('adjacency_matrix.csv')
    return adj_df

def prepare_data(opt):
    dataset = {}
# 782-miRNA-cancer数据构建
    mi_dis_data = association_to_matrix('')

    #返回邻接矩阵的行和列索引 用于生成带有序号的特征表示
    # 获取DataFrame的行索引
    row_indexes = mi_dis_data.index
    # 获取DataFrame的列索引
    column_indexes = mi_dis_data.columns
    dataset['row_indexes_MDA'] = row_indexes
    dataset['column_indexes_MDA'] = column_indexes



    # 邻接矩阵读取
    # mi_dis_data = pd.read_csv('adjacency_matrix.csv', index_col = 0)
    # md_p和md_true都为邻接矩阵 格式不同
    dataset['md_p'] = t.FloatTensor(np.array(mi_dis_data).T)
    dataset['md_true'] = dataset['md_p']
    # 查找邻接矩阵中值为0和1的位置 存放在两个list中
    all_zero_index = []
    all_one_index = []
    for i in range(dataset['md_p'].size(0)) :
        for j in range(dataset['md_p'].size(1)) :
            if dataset['md_p'][i][j] < 1 :
                all_zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1 :
                all_one_index.append([i, j])
    # 打乱顺序
    np.random.seed(0)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)

    dataset['md'] = []
    # 将全部的零样本索引和一样本索引转换为 PyTorch 的 LongTensor 类型
    zero_tensor = t.LongTensor(all_zero_index)
    one_tensor = t.LongTensor(all_one_index)

    # 将全部样本的索引作为训练集索引
    train_zero_index = zero_tensor
    train_one_index = one_tensor

    dataset['md'].append({'test' : [train_one_index, train_zero_index],
                          'train' : [train_one_index, train_zero_index]})


    print('计算高斯核相似性')
    DGSM = Gauss_D(dataset['md_p'].numpy(), dataset['md_p'].size(1))
    MGSM = Gauss_M(dataset['md_p'].numpy(), dataset['md_p'].size(0))

    print(DGSM.shape)
    print(MGSM.shape)

    print('计算sigmoid核相似性')

    # 计算sigmoid相似性 计算后保存可使用读取 节省时间
    # Sigmoid_matrix1 = SigmoidKernelDisease(dataset['md_p'].numpy())
    # Sigmoid_matrix2 = SigmoidKernelRNA(dataset['md_p'].numpy())
    # storFile(Sigmoid_matrix1, 'MDA_mi_sigmoid.csv')
    # storFile(Sigmoid_matrix2, 'MDA_dis_sigmoid.csv')
    Sigmoid_matrix1 = pd.read_csv('MDA_mi_sigmoid.csv', header=None, index_col=None)
    Sigmoid_matrix2 = pd.read_csv('MDA_dis_sigmoid.csv', header=None, index_col=None)
    Sigmoid_matrix1 = np.array(Sigmoid_matrix1)
    Sigmoid_matrix2 = np.array(Sigmoid_matrix2)
    print(Sigmoid_matrix1.shape)
    print(Sigmoid_matrix2.shape)


    # 两种矩阵融合
    nd = mi_dis_data.shape[0]
    nm = mi_dis_data.shape[1]

    print('nd大小为', nd)
    print('nm大小为',nm)

    ID = np.zeros([nd, nd])
    # 两种相似性相加除以二
    # 高斯核以外的另一种相似性为文件导入
    # 高斯核相似性采用自定义方法计算
    for h1 in range(nd) :
        for h2 in range(nd) :
            if Sigmoid_matrix2[h1, h2] == 0 :
                ID[h1, h2] = DGSM[h1, h2]
            else :
                ID[h1, h2] = (Sigmoid_matrix2[h1, h2] + DGSM[h1, h2]) / 2

    IM = np.zeros([nm, nm])

    for q1 in range(nm) :
        for q2 in range(nm) :
            if Sigmoid_matrix1[q1, q2] == 0 :
                IM[q1, q2] = MGSM[q1, q2]
            else :
                IM[q1, q2] = (Sigmoid_matrix1[q1, q2] + MGSM[q1, q2]) / 2

    dataset['ID'] = t.from_numpy(ID)
    dataset['IM'] = t.from_numpy(IM)


    # 计算walker
    Walker_Mi = pd.read_csv('MDA_mi_Walker.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('MDA_disease_Walker.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['MDA_SD_OD'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['MDA_SM_OD'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['MDA_SD_S'] = t.from_numpy(np.array(Walker_Dis))
    dataset['MDA_SM_S'] = t.from_numpy(np.array(Walker_Mi))

    # 计算ST
    Walker_Mi_ST = pd.read_csv('MDA_mi_ST.csv', header = None, index_col = None)
    Walker_Dis_ST = pd.read_csv('MDA_disease_ST.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis_ST = np.array(Eu_dis(Walker_Mi_ST))
    Walker_Dis_Eu_dis_ST = np.array(Eu_dis(Walker_Dis_ST))

    dataset['MDA_SD_OD_ST'] = t.from_numpy(Walker_Dis_Eu_dis_ST)
    dataset['MDA_SM_OD_ST'] = t.from_numpy(Walker_Mi_Eu_dis_ST)

    dataset['MDA_SD_S_ST'] = t.from_numpy(np.array(Walker_Dis_ST))
    dataset['MDA_SM_S_ST'] = t.from_numpy(np.array(Walker_Mi_ST))
#

    circ_dis_data = association_to_matrix('')

    #返回邻接矩阵的行和列索引 用于生成带有序号的特征表示
    # 获取DataFrame的行索引
    row_indexes = circ_dis_data.index
    # 获取DataFrame的列索引
    column_indexes = circ_dis_data.columns
    dataset['row_indexes_CDA'] = row_indexes
    dataset['column_indexes_CDA'] = column_indexes

    # 邻接矩阵读取
    # mi_dis_data = pd.read_csv('adjacency_matrix.csv', index_col = 0)
    # md_p和md_true都为邻接矩阵 格式不同
    dataset['cd_p'] = t.FloatTensor(np.array(circ_dis_data).T)
    dataset['cd_true'] = dataset['cd_p']
    # 查找邻接矩阵中值为0和1的位置 存放在两个list中
    all_zero_index = []
    all_one_index = []
    for i in range(dataset['cd_p'].size(0)) :
        for j in range(dataset['cd_p'].size(1)) :
            if dataset['cd_p'][i][j] < 1 :
                all_zero_index.append([i, j])
            if dataset['cd_p'][i][j] >= 1 :
                all_one_index.append([i, j])
    # 打乱顺序
    np.random.seed(0)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)

    dataset['cd'] = []
    # 将全部的零样本索引和一样本索引转换为 PyTorch 的 LongTensor 类型
    zero_tensor = t.LongTensor(all_zero_index)
    one_tensor = t.LongTensor(all_one_index)

    # 将全部样本的索引作为训练集索引
    train_zero_index = zero_tensor
    train_one_index = one_tensor

    dataset['cd'].append({'test' : [train_one_index, train_zero_index],
                          'train' : [train_one_index, train_zero_index]})


    print('计算高斯核相似性')
    DGSM = Gauss_D(dataset['cd_p'].numpy(), dataset['cd_p'].size(1))
    MGSM = Gauss_M(dataset['cd_p'].numpy(), dataset['cd_p'].size(0))

    print(DGSM.shape)
    print(MGSM.shape)

    print('计算sigmoid核相似性')


    Sigmoid_matrix3 = pd.read_csv('CDA_circ_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix4 = pd.read_csv('CDA_dis_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix1 = np.array(Sigmoid_matrix3)
    Sigmoid_matrix2 = np.array(Sigmoid_matrix4)
    print(Sigmoid_matrix1.shape)
    print(Sigmoid_matrix2.shape)


    # 两种矩阵融合
    nd = circ_dis_data.shape[0]
    nm = circ_dis_data.shape[1]

    print('nd大小为', nd)
    print('nm大小为', nm)

    ID = np.zeros([nd, nd])
    # 两种相似性相加除以二
    # 高斯核以外的另一种相似性为文件导入
    # 高斯核相似性采用自定义方法计算
    for h1 in range(nd) :
        for h2 in range(nd) :
            if Sigmoid_matrix2[h1, h2] == 0 :
                ID[h1, h2] = DGSM[h1, h2]
            else :
                ID[h1, h2] = (Sigmoid_matrix2[h1, h2] + DGSM[h1, h2]) / 2

    IM = np.zeros([nm, nm])

    for q1 in range(nm) :
        for q2 in range(nm) :
            if Sigmoid_matrix1[q1, q2] == 0 :
                IM[q1, q2] = MGSM[q1, q2]
            else :
                IM[q1, q2] = (Sigmoid_matrix1[q1, q2] + MGSM[q1, q2]) / 2

    dataset['CDA_ID'] = t.from_numpy(ID)
    dataset['CDA_IM'] = t.from_numpy(IM)

    # 计算walker
    Walker_Mi = pd.read_csv('CDA_circ_Walker.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('CDA_disease_Walker.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['CDA_SD_OD'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['CDA_SM_OD'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['CDA_SD_S'] = t.from_numpy(np.array(Walker_Dis))
    dataset['CDA_SM_S'] = t.from_numpy(np.array(Walker_Mi))

    # 计算ST
    Walker_Mi_ST = pd.read_csv('CDA_circ_ST.csv', header = None, index_col = None)
    Walker_Dis_ST = pd.read_csv('CDA_disease_ST.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis_ST = np.array(Eu_dis(Walker_Mi_ST))
    Walker_Dis_Eu_dis_ST = np.array(Eu_dis(Walker_Dis_ST))

    dataset['CDA_SD_OD_ST'] = t.from_numpy(Walker_Dis_Eu_dis_ST)
    dataset['CDA_SM_OD_ST'] = t.from_numpy(Walker_Mi_Eu_dis_ST)

    dataset['CDA_SD_S_ST'] = t.from_numpy(np.array(Walker_Dis_ST))
    dataset['CDA_SM_S_ST'] = t.from_numpy(np.array(Walker_Mi_ST))
#

    lnc_dis_data = association_to_matrix('')

    #返回邻接矩阵的行和列索引 用于生成带有序号的特征表示
    # 获取DataFrame的行索引
    row_indexes = lnc_dis_data.index
    # 获取DataFrame的列索引
    column_indexes = lnc_dis_data.columns
    dataset['row_indexes_LDA'] = row_indexes
    dataset['column_indexes_LDA'] = column_indexes

    # 邻接矩阵读取
    # mi_dis_data = pd.read_csv('adjacency_matrix.csv', index_col = 0)
    # md_p和md_true都为邻接矩阵 格式不同
    dataset['ld_p'] = t.FloatTensor(np.array(lnc_dis_data).T)
    dataset['ld_true'] = dataset['ld_p']
    # 查找邻接矩阵中值为0和1的位置 存放在两个list中
    all_zero_index = []
    all_one_index = []
    for i in range(dataset['ld_p'].size(0)) :
        for j in range(dataset['ld_p'].size(1)) :
            if dataset['ld_p'][i][j] < 1 :
                all_zero_index.append([i, j])
            if dataset['ld_p'][i][j] >= 1 :
                all_one_index.append([i, j])
    # 打乱顺序
    np.random.seed(0)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)

    dataset['ld'] = []
    # 将全部的零样本索引和一样本索引转换为 PyTorch 的 LongTensor 类型
    zero_tensor = t.LongTensor(all_zero_index)
    one_tensor = t.LongTensor(all_one_index)

    # 将全部样本的索引作为训练集索引
    train_zero_index = zero_tensor
    train_one_index = one_tensor

    dataset['ld'].append({'test' : [train_one_index, train_zero_index],
                          'train' : [train_one_index, train_zero_index]})


    print('计算高斯核相似性')
    DGSM = Gauss_D(dataset['ld_p'].numpy(), dataset['ld_p'].size(1))
    MGSM = Gauss_M(dataset['ld_p'].numpy(), dataset['ld_p'].size(0))

    print(DGSM.shape)
    print(MGSM.shape)

    print('计算sigmoid核相似性')


    Sigmoid_matrix3 = pd.read_csv('LDA_lnc_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix4 = pd.read_csv('LDA_dis_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix1 = np.array(Sigmoid_matrix3)
    Sigmoid_matrix2 = np.array(Sigmoid_matrix4)
    print(Sigmoid_matrix1.shape)
    print(Sigmoid_matrix2.shape)

    # 两种矩阵融合
    nd = lnc_dis_data.shape[0]
    nm = lnc_dis_data.shape[1]

    print('nd大小为', nd)
    print('nm大小为', nm)

    ID = np.zeros([nd, nd])
    # 两种相似性相加除以二
    # 高斯核以外的另一种相似性为文件导入
    # 高斯核相似性采用自定义方法计算
    for h1 in range(nd) :
        for h2 in range(nd) :
            if Sigmoid_matrix2[h1, h2] == 0 :
                ID[h1, h2] = DGSM[h1, h2]
            else :
                ID[h1, h2] = (Sigmoid_matrix2[h1, h2] + DGSM[h1, h2]) / 2

    IM = np.zeros([nm, nm])

    for q1 in range(nm) :
        for q2 in range(nm) :
            if Sigmoid_matrix1[q1, q2] == 0 :
                IM[q1, q2] = MGSM[q1, q2]
            else :
                IM[q1, q2] = (Sigmoid_matrix1[q1, q2] + MGSM[q1, q2]) / 2

    dataset['LDA_ID'] = t.from_numpy(ID)
    dataset['LDA_IM'] = t.from_numpy(IM)

    # 计算walker
    Walker_Mi = pd.read_csv('LDA_lnc_Walker.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('LDA_disease_Walker.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['LDA_SD_OD'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['LDA_SM_OD'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['LDA_SD_S'] = t.from_numpy(np.array(Walker_Dis))
    dataset['LDA_SM_S'] = t.from_numpy(np.array(Walker_Mi))

    # 计算ST
    Walker_Mi = pd.read_csv('LDA_lnc_ST.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('LDA_disease_ST.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['LDA_SD_OD_ST'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['LDA_SM_OD_ST'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['LDA_SD_S_ST'] = t.from_numpy(np.array(Walker_Dis))
    dataset['LDA_SM_S_ST'] = t.from_numpy(np.array(Walker_Mi))




    mi_gene_data = association_to_matrix('')

    #返回邻接矩阵的行和列索引 用于生成带有序号的特征表示
    # 获取DataFrame的行索引
    row_indexes = mi_gene_data.index
    # 获取DataFrame的列索引
    column_indexes = mi_gene_data.columns
    dataset['row_indexes_MGA'] = row_indexes
    dataset['column_indexes_MGA'] = column_indexes

    # 邻接矩阵读取
    # mi_dis_data = pd.read_csv('adjacency_matrix.csv', index_col = 0)
    # md_p和md_true都为邻接矩阵 格式不同
    dataset['mg_p'] = t.FloatTensor(np.array(mi_gene_data))
    dataset['mg_true'] = dataset['mg_p']
    # 查找邻接矩阵中值为0和1的位置 存放在两个list中
    all_zero_index = []
    all_one_index = []
    for i in range(dataset['mg_p'].size(0)) :
        for j in range(dataset['mg_p'].size(1)) :
            if dataset['mg_p'][i][j] < 1 :
                all_zero_index.append([i, j])
            if dataset['mg_p'][i][j] >= 1 :
                all_one_index.append([i, j])
    # 打乱顺序
    np.random.seed(0)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)

    dataset['mg'] = []
    # 将全部的零样本索引和一样本索引转换为 PyTorch 的 LongTensor 类型
    zero_tensor = t.LongTensor(all_zero_index)
    one_tensor = t.LongTensor(all_one_index)

    # 将全部样本的索引作为训练集索引
    train_zero_index = zero_tensor
    train_one_index = one_tensor

    dataset['mg'].append({'test' : [train_one_index, train_zero_index],
                          'train' : [train_one_index, train_zero_index]})


    print('计算高斯核相似性')
    DGSM = Gauss_D(dataset['mg_p'].numpy(), dataset['mg_p'].size(1))
    MGSM = Gauss_M(dataset['mg_p'].numpy(), dataset['mg_p'].size(0))

    print(DGSM.shape)
    print(MGSM.shape)

    print('计算sigmoid核相似性')

    Sigmoid_matrix3 = pd.read_csv('MGA_mi_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix4 = pd.read_csv('MGA_gene_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix1 = np.array(Sigmoid_matrix3)
    Sigmoid_matrix2 = np.array(Sigmoid_matrix4)
    print(Sigmoid_matrix1.shape)
    print(Sigmoid_matrix2.shape)

    # 两种矩阵融合
    nd = mi_gene_data.shape[0]
    nm = mi_gene_data.shape[1]

    print('nd大小为', nd)
    print('nm大小为', nm)

    ID = np.zeros([nd, nd])
    # 两种相似性相加除以二
    # 高斯核以外的另一种相似性为文件导入
    # 高斯核相似性采用自定义方法计算
    for h1 in range(nd) :
        for h2 in range(nd) :
            if Sigmoid_matrix1[h1, h2] == 0 :
                ID[h1, h2] = MGSM[h1, h2]
            else :
                ID[h1, h2] = (Sigmoid_matrix1[h1, h2] + MGSM[h1, h2]) / 2

    IM = np.zeros([nm, nm])

    for q1 in range(nm) :
        for q2 in range(nm) :
            if Sigmoid_matrix2[q1, q2] == 0 :
                IM[q1, q2] = DGSM[q1, q2]
            else :
                IM[q1, q2] = (Sigmoid_matrix2[q1, q2] + DGSM[q1, q2]) / 2

    dataset['MGA_ID'] = t.from_numpy(ID)
    dataset['MGA_IM'] = t.from_numpy(IM)

    # 计算walker
    Walker_Mi = pd.read_csv('MGA_gene_Walker.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('MGA_mi_Walker.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['MGA_SD_OD'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['MGA_SM_OD'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['MGA_SD_S'] = t.from_numpy(np.array(Walker_Dis))
    dataset['MGA_SM_S'] = t.from_numpy(np.array(Walker_Mi))

    # 计算ST
    Walker_Mi = pd.read_csv('MGA_gene_ST.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('MGA_mi_ST.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['MGA_SD_OD_ST'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['MGA_SM_OD_ST'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['MGA_SD_S_ST'] = t.from_numpy(np.array(Walker_Dis))
    dataset['MGA_SM_S_ST'] = t.from_numpy(np.array(Walker_Mi))



    circ_mi_data = association_to_matrix('')

    #返回邻接矩阵的行和列索引 用于生成带有序号的特征表示
    # 获取DataFrame的行索引
    row_indexes = circ_mi_data.index
    # 获取DataFrame的列索引
    column_indexes = circ_mi_data.columns
    dataset['row_indexes_CMI'] = row_indexes
    dataset['column_indexes_CMI'] = column_indexes

    # 邻接矩阵读取
    # mi_dis_data = pd.read_csv('adjacency_matrix.csv', index_col = 0)
    # md_p和md_true都为邻接矩阵 格式不同
    dataset['cm_p'] = t.FloatTensor(np.array(circ_mi_data).T)
    dataset['cm_true'] = dataset['cm_p']
    # 查找邻接矩阵中值为0和1的位置 存放在两个list中
    all_zero_index = []
    all_one_index = []
    for i in range(dataset['cm_p'].size(0)) :
        for j in range(dataset['cm_p'].size(1)) :
            if dataset['cm_p'][i][j] < 1 :
                all_zero_index.append([i, j])
            if dataset['cm_p'][i][j] >= 1 :
                all_one_index.append([i, j])
    # 打乱顺序
    np.random.seed(0)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)

    dataset['cm'] = []
    # 将全部的零样本索引和一样本索引转换为 PyTorch 的 LongTensor 类型
    zero_tensor = t.LongTensor(all_zero_index)
    one_tensor = t.LongTensor(all_one_index)

    # 将全部样本的索引作为训练集索引
    train_zero_index = zero_tensor
    train_one_index = one_tensor

    dataset['cm'].append({'test' : [train_one_index, train_zero_index],
                          'train' : [train_one_index, train_zero_index]})


    print('计算高斯核相似性')
    DGSM = Gauss_D(dataset['cm_p'].numpy(), dataset['cm_p'].size(1))
    MGSM = Gauss_M(dataset['cm_p'].numpy(), dataset['cm_p'].size(0))

    print(DGSM.shape)
    print(MGSM.shape)

    print('计算sigmoid核相似性')


    Sigmoid_matrix3 = pd.read_csv('CMI_circ_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix4 = pd.read_csv('CMI_mi_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix1 = np.array(Sigmoid_matrix3)
    Sigmoid_matrix2 = np.array(Sigmoid_matrix4)
    print(Sigmoid_matrix1.shape)
    print(Sigmoid_matrix2.shape)

    # 两种矩阵融合
    nd = circ_mi_data.shape[0]
    nm = circ_mi_data.shape[1]

    print('nd大小为', nd)
    print('nm大小为', nm)

    ID = np.zeros([nd, nd])
    # 两种相似性相加除以二
    # 高斯核以外的另一种相似性为文件导入
    # 高斯核相似性采用自定义方法计算
    for h1 in range(nd) :
        for h2 in range(nd) :
            if Sigmoid_matrix2[h1, h2] == 0 :
                ID[h1, h2] = DGSM[h1, h2]
            else :
                ID[h1, h2] = (Sigmoid_matrix2[h1, h2] + DGSM[h1, h2]) / 2

    IM = np.zeros([nm, nm])

    for q1 in range(nm) :
        for q2 in range(nm) :
            if Sigmoid_matrix1[q1, q2] == 0 :
                IM[q1, q2] = MGSM[q1, q2]
            else :
                IM[q1, q2] = (Sigmoid_matrix1[q1, q2] + MGSM[q1, q2]) / 2

    dataset['CMI_ID'] = t.from_numpy(ID)
    dataset['CMI_IM'] = t.from_numpy(IM)

    # 计算walker
    Walker_Mi = pd.read_csv('CMI_circ_Walker.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('CMI_mi_Walker.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['CMI_SD_OD'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['CMI_SM_OD'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['CMI_SD_S'] = t.from_numpy(np.array(Walker_Dis))
    dataset['CMI_SM_S'] = t.from_numpy(np.array(Walker_Mi))

    # 计算ST
    Walker_Mi = pd.read_csv('CMI_circ_ST.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('CMI_mi_ST.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['CMI_SD_OD_ST'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['CMI_SM_OD_ST'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['CMI_SD_S_ST'] = t.from_numpy(np.array(Walker_Dis))
    dataset['CMI_SM_S_ST'] = t.from_numpy(np.array(Walker_Mi))


    lnc_mi_data = association_to_matrix('')

    #返回邻接矩阵的行和列索引 用于生成带有序号的特征表示
    # 获取DataFrame的行索引
    row_indexes = lnc_mi_data.index
    # 获取DataFrame的列索引
    column_indexes = lnc_mi_data.columns
    dataset['row_indexes_LMI'] = row_indexes
    dataset['column_indexes_LMI'] = column_indexes

    # 邻接矩阵读取
    # mi_dis_data = pd.read_csv('adjacency_matrix.csv', index_col = 0)
    # md_p和md_true都为邻接矩阵 格式不同
    dataset['lm_p'] = t.FloatTensor(np.array(lnc_mi_data))
    dataset['lm_true'] = dataset['lm_p']
    # 查找邻接矩阵中值为0和1的位置 存放在两个list中
    all_zero_index = []
    all_one_index = []
    for i in range(dataset['lm_p'].size(0)) :
        for j in range(dataset['lm_p'].size(1)) :
            if dataset['lm_p'][i][j] < 1 :
                all_zero_index.append([i, j])
            if dataset['lm_p'][i][j] >= 1 :
                all_one_index.append([i, j])
    # 打乱顺序
    np.random.seed(0)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)

    dataset['lm'] = []
    # 将全部的零样本索引和一样本索引转换为 PyTorch 的 LongTensor 类型
    zero_tensor = t.LongTensor(all_zero_index)
    one_tensor = t.LongTensor(all_one_index)

    # 将全部样本的索引作为训练集索引
    train_zero_index = zero_tensor
    train_one_index = one_tensor

    dataset['lm'].append({'test' : [train_one_index, train_zero_index],
                          'train' : [train_one_index, train_zero_index]})

    print('计算高斯核相似性')
    DGSM = Gauss_D(dataset['lm_p'].numpy(), dataset['lm_p'].size(1))
    MGSM = Gauss_M(dataset['lm_p'].numpy(), dataset['lm_p'].size(0))

    print(DGSM.shape)
    print(MGSM.shape)

    print('计算sigmoid核相似性')

    Sigmoid_matrix3 = pd.read_csv('LMI_lnc_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix4 = pd.read_csv('LMI_mi_sigmoid.csv', header = None, index_col = None)
    Sigmoid_matrix1 = np.array(Sigmoid_matrix3)
    Sigmoid_matrix2 = np.array(Sigmoid_matrix4)
    print(Sigmoid_matrix1.shape)
    print(Sigmoid_matrix2.shape)

    # 两种矩阵融合
    nd = lnc_mi_data.shape[0]
    nm = lnc_mi_data.shape[1]

    print('nd大小为', nd)
    print('nm大小为', nm)

    ID = np.zeros([nd, nd])
    # 两种相似性相加除以二
    # 高斯核以外的另一种相似性为文件导入
    # 高斯核相似性采用自定义方法计算
    for h1 in range(nd) :
        for h2 in range(nd) :
            if Sigmoid_matrix1[h1, h2] == 0 :
                ID[h1, h2] = MGSM[h1, h2]
            else :
                ID[h1, h2] = (Sigmoid_matrix1[h1, h2] + MGSM[h1, h2]) / 2

    IM = np.zeros([nm, nm])

    for q1 in range(nm) :
        for q2 in range(nm) :
            if Sigmoid_matrix2[q1, q2] == 0 :
                IM[q1, q2] = DGSM[q1, q2]
            else :
                IM[q1, q2] = (Sigmoid_matrix2[q1, q2] + DGSM[q1, q2]) / 2

    dataset['LMI_ID'] = t.from_numpy(ID)
    dataset['LMI_IM'] = t.from_numpy(IM)

    # 计算walker
    Walker_Mi = pd.read_csv('LMI_mi_Walker.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('LMI_lnc_Walker.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['LMI_SD_OD'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['LMI_SM_OD'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['LMI_SD_S'] = t.from_numpy(np.array(Walker_Dis))
    dataset['LMI_SM_S'] = t.from_numpy(np.array(Walker_Mi))

    # 计算ST
    Walker_Mi = pd.read_csv('LMI_mi_ST.csv', header = None, index_col = None)
    Walker_Dis = pd.read_csv('LMI_lnc_ST.csv', header = None, index_col = None)
    # 计算walker欧氏距离
    Walker_Mi_Eu_dis = np.array(Eu_dis(Walker_Mi))
    Walker_Dis_Eu_dis = np.array(Eu_dis(Walker_Dis))

    dataset['LMI_SD_OD_ST'] = t.from_numpy(Walker_Dis_Eu_dis)
    dataset['LMI_SM_OD_ST'] = t.from_numpy(Walker_Mi_Eu_dis)

    dataset['LMI_SD_S_ST'] = t.from_numpy(np.array(Walker_Dis))
    dataset['LMI_SM_S_ST'] = t.from_numpy(np.array(Walker_Mi))

    return dataset
