import pandas as pd
import torch
from prepare_data import prepare_data
import numpy as np
from torch import optim
from param import parameter_parser
from Module_demo import MMLMGNN
from utils import get_L2reg, Myloss
from trainData import Dataset_MDA
import ConstructHW



import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, train_data, optim, opt):

    model.train()
    regression_crit = Myloss()

    one_index = train_data[2][0]
    zero_index = train_data[2][1]


    dis_sim_integrate_tensor = train_data[0].to(device)
    mi_sim_integrate_tensor = train_data[1].to(device)

    concat_miRNA = np.hstack(
        [train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
    concat_mi_tensor = torch.FloatTensor(concat_miRNA)
    concat_mi_tensor = concat_mi_tensor.to(device)

    G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.detach().cpu().numpy(), K_neigs=[13], is_probH=False)
    G_mi_Km = ConstructHW.constructHW_kmean(concat_mi_tensor.detach().cpu().numpy(), clusters=[9])
    G_mi_Kn = G_mi_Kn.to(device)
    G_mi_Km = G_mi_Km.to(device)


    concat_dis = np.hstack(
        [train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
    concat_dis_tensor = torch.FloatTensor(concat_dis)
    concat_dis_tensor = concat_dis_tensor.to(device)


    G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.detach().cpu().numpy(), K_neigs=[13], is_probH=False)
    G_dis_Km = ConstructHW.constructHW_kmean(concat_dis_tensor.detach().cpu().numpy(), clusters=[9])
    G_dis_Kn = G_dis_Kn.to(device)
    G_dis_Km = G_dis_Km.to(device)

    X1_list = []
    Y1_list = []
    X1_knn_list = []
    Y1_knn_list = []
    X1_km_list = []
    Y1_km_list = []


    for epoch in range(1, opt.epoch+1):
        score, mi_cl_loss, dis_cl_loss, X1, Y1, mi_feature1, mi_feature2, dis_feature1, dis_feature2, mi_concat_feature, dis_concat_feature\
            = model(concat_mi_tensor, concat_dis_tensor,
                    G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km,
                                            )
        X1_np = X1.detach().cpu().numpy()
        X1_list.append(X1_np)
        print(X1_np)
        X1_df = pd.DataFrame(X1_np)
        X1_df.to_csv(f'SM_X1_全连接降维_epoch_{epoch}.csv', header = None, index = None)

        Y1_np = Y1.detach().cpu().numpy()

        Y1_list.append(Y1_np)

        print(Y1_np)

        Y1_df = pd.DataFrame(Y1_np)
        Y1_df.to_csv(f'SM_Y1_全连接降维_epoch_{epoch}.csv', header = None, index = None)

        X1_knn = mi_feature1.detach().cpu().numpy()

        X1_knn_list.append(X1_knn)

        print(X1_knn)

        X1_knn_df = pd.DataFrame(X1_knn)
        X1_knn_df.to_csv(f'SM_X1_knn_epoch_{epoch}.csv', header = None, index = None)

        Y1_knn = dis_feature1.detach().cpu().numpy()

        Y1_knn_list.append(Y1_knn)

        print(Y1_knn)

        Y1_knn_df = pd.DataFrame(Y1_knn)
        Y1_knn_df.to_csv(f'SM_Y1_knn_epoch_{epoch}.csv', header = None, index = None)

        X1_km = mi_feature2.detach().cpu().numpy()

        X1_km_list.append(X1_km)

        print(X1_km)

        X1_km_df = pd.DataFrame(X1_km)
        X1_km_df.to_csv(f'SM_X1_km_epoch_{epoch}.csv', header = None, index = None)

        Y1_km = dis_feature2.detach().cpu().numpy()

        Y1_km_list.append(Y1_km)

        print(Y1_km)

        Y1_km_df = pd.DataFrame(Y1_km)
        Y1_km_df.to_csv(f'SM_Y1_km_epoch_{epoch}.csv', header = None, index = None)



        recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
        reg_loss = get_L2reg(model.parameters())

        tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00001 * reg_loss
        optim.zero_grad()
        tol_loss.backward()
        optim.step()

    true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(model, train_data, concat_mi_tensor, concat_dis_tensor,
                                 G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)

    return true_value_one, true_value_zero, pre_value_one, pre_value_zero

def train_epoch_MDA_ST(model, train_data, optim, opt) :

    model.train()
    regression_crit = Myloss()

    one_index = train_data[2][0]
    zero_index = train_data[2][1]



    dis_Walker_integrate_tensor = train_data[14].to(device)
    mi_Walker_integrate_tensor = train_data[15].to(device)

    dis_Walker_OD_tensor = train_data[12].to(device)
    mi_Walker_OD_tensor = train_data[13].to(device)

    mi_Walker_integrate_tensor1 = np.hstack(
        [train_data[4].numpy(), mi_Walker_integrate_tensor.detach().cpu().numpy()])
    Walker_mi_tensor = torch.FloatTensor(mi_Walker_integrate_tensor1)
    Walker_mi_tensor = Walker_mi_tensor.to(device)

    G_Walker_mi_Kn = ConstructHW.constructHW_knn(Walker_mi_tensor.detach().cpu().numpy(), K_neigs = [13],
                                                 is_probH = False)
    G_Walker_mi_Km = ConstructHW.constructHW_kmean(Walker_mi_tensor.detach().cpu().numpy(), clusters = [9])
    G_Walker_mi_Kn = G_Walker_mi_Kn.to(device)
    G_Walker_mi_Km = G_Walker_mi_Km.to(device)


    dis_Walker_integrate_tensor1 = np.hstack(
        [train_data[4].numpy().T, dis_Walker_integrate_tensor.detach().cpu().numpy()])
    Walker_dis_tensor = torch.FloatTensor(dis_Walker_integrate_tensor1)
    Walker_dis_tensor = Walker_dis_tensor.to(device)

    G_Walker_dis_Kn = ConstructHW.constructHW_knn(Walker_dis_tensor.detach().cpu().numpy(), K_neigs = [13],
                                                  is_probH = False)
    G_Walker_dis_Km = ConstructHW.constructHW_kmean(Walker_dis_tensor.detach().cpu().numpy(), clusters = [9])
    G_Walker_dis_Kn = G_Walker_dis_Kn.to(device)
    G_Walker_dis_Km = G_Walker_dis_Km.to(device)


    mi_Walker_OD_tensor1 = np.hstack(
        [train_data[4].numpy(), mi_Walker_OD_tensor.detach().cpu().numpy()])
    Walker_mi_OD_tensor = torch.FloatTensor(mi_Walker_OD_tensor1)
    Walker_mi_OD_tensor = Walker_mi_OD_tensor.to(device)

    dis_Walker_OD_tensor1 = np.hstack(
        [train_data[4].numpy().T, dis_Walker_OD_tensor.detach().cpu().numpy()])
    Walker_dis_OD_tensor = torch.FloatTensor(dis_Walker_OD_tensor1)
    Walker_dis_OD_tensor = Walker_dis_OD_tensor.to(device)



    TP_X1_list = []
    TP_Y1_list = []
    TP_X1_knn_list = []
    TP_Y1_knn_list = []
    TP_X1_km_list = []
    TP_Y1_km_list = []


    for epoch in range(1, opt.epoch + 1) :

        score_TP, mi_TP_loss, dis_TP_loss, TP_X1, TP_Y1\
            , TP_mi_feature1, TP_mi_feature2, TP_dis_feature1, TP_dis_feature2, TP_mi_concat_feature, TP_dis_concat_feature\
            = model(Walker_mi_OD_tensor, Walker_dis_OD_tensor,
                                                              G_Walker_mi_Kn, G_Walker_mi_Km, G_Walker_dis_Kn,
                                                              G_Walker_dis_Km)

        TP_X1_np = TP_X1.detach().cpu().numpy()

        TP_X1_list.append(TP_X1_np)

        print(TP_X1_np)

        TP_X1_df = pd.DataFrame(TP_X1_np)

        index1 = pd.DataFrame(train_data[7])
        TP_X1_df = pd.concat([index1, TP_X1_df], axis = 1)
        TP_X1_df.to_csv(f'MDA_ST_mi_全连接降维_epoch_{epoch}.csv', header = None, index = None)

        TP_Y1_np = TP_Y1.detach().cpu().numpy()

        TP_Y1_list.append(TP_Y1_np)

        print(TP_Y1_np)

        TP_Y1_df = pd.DataFrame(TP_Y1_np)
        index2 = pd.DataFrame(train_data[6])
        TP_Y1_df = pd.concat([index2, TP_Y1_df], axis = 1)
        TP_Y1_df.to_csv(f'MDA_ST_dis_全连接降维_epoch_{epoch}.csv', header = None, index = None)

        TP_X1_knn = TP_mi_feature1.detach().cpu().numpy()

        TP_X1_knn_list.append(TP_X1_knn)

        print(TP_X1_knn)

        TP_X1_knn_df = pd.DataFrame(TP_X1_knn)
        index1 = pd.DataFrame(train_data[7])
        TP_X1_knn_df = pd.concat([index1, TP_X1_knn_df], axis = 1)
        TP_X1_knn_df.to_csv(f'MDA_ST_mi_knn_epoch_{epoch}.csv', header = None, index = None)

        TP_Y1_knn = TP_dis_feature1.detach().cpu().numpy()

        TP_Y1_knn_list.append(TP_Y1_knn)

        print(TP_Y1_knn)

        TP_Y1_knn_df = pd.DataFrame(TP_Y1_knn)
        index2 = pd.DataFrame(train_data[6])
        TP_Y1_knn_df = pd.concat([index2, TP_Y1_knn_df], axis = 1)
        TP_Y1_knn_df.to_csv(f'MDA_ST_dis_knn_epoch_{epoch}.csv', header = None, index = None)

        TP_X1_km = TP_mi_feature2.detach().cpu().numpy()

        TP_X1_km_list.append(TP_X1_km)

        print(TP_X1_km)

        TP_X1_km_df = pd.DataFrame(TP_X1_km)
        index1 = pd.DataFrame(train_data[7])
        TP_X1_km_df = pd.concat([index1, TP_X1_km_df], axis = 1)
        TP_X1_km_df.to_csv(f'MDA_ST_mi_km_epoch_{epoch}.csv', header = None, index = None)

        TP_Y1_km = TP_dis_feature2.detach().cpu().numpy()

        TP_Y1_km_list.append(TP_Y1_km)

        print(TP_Y1_km)

        TP_Y1_km_df = pd.DataFrame(TP_Y1_km)
        index2 = pd.DataFrame(train_data[6])
        TP_Y1_km_df = pd.concat([index2, TP_Y1_km_df], axis = 1)
        TP_Y1_km_df.to_csv(f'MDA_ST_dis_km_epoch_{epoch}.csv', header = None, index = None)

        recover_loss_TP = regression_crit(one_index, zero_index, train_data[4].to(device), score_TP)
        reg_loss_TP = get_L2reg(model.parameters())

        tol_loss_TP = recover_loss_TP + mi_TP_loss + dis_TP_loss + 0.00001 * reg_loss_TP
        optim.zero_grad()
        tol_loss_TP.backward()
        optim.step()

    true_value_one_TP, true_value_zero_TP, pre_value_one_TP, pre_value_zero_TP = test(model, train_data, Walker_mi_OD_tensor, Walker_dis_OD_tensor,
                                                              G_Walker_mi_Kn, G_Walker_mi_Km, G_Walker_dis_Kn,
                                                              G_Walker_dis_Km)

    return true_value_one_TP, true_value_zero_TP, pre_value_one_TP, pre_value_zero_TP

def train_epoch_MDA_TP(model, train_data, optim, opt) :

    model.train()
    regression_crit = Myloss()

    one_index = train_data[2][0]
    zero_index = train_data[2][1]
    # print(train_data)


    dis_Walker_integrate_tensor = train_data[10].to(device)
    mi_Walker_integrate_tensor = train_data[11].to(device)

    dis_Walker_OD_tensor = train_data[8].to(device)
    mi_Walker_OD_tensor = train_data[9].to(device)

    mi_Walker_integrate_tensor1 = np.hstack(
        [train_data[4].numpy(), mi_Walker_integrate_tensor.detach().cpu().numpy()])
    Walker_mi_tensor = torch.FloatTensor(mi_Walker_integrate_tensor1)
    Walker_mi_tensor = Walker_mi_tensor.to(device)

    G_Walker_mi_Kn = ConstructHW.constructHW_knn(Walker_mi_tensor.detach().cpu().numpy(), K_neigs = [13],
                                                 is_probH = False)
    G_Walker_mi_Km = ConstructHW.constructHW_kmean(Walker_mi_tensor.detach().cpu().numpy(), clusters = [9])
    G_Walker_mi_Kn = G_Walker_mi_Kn.to(device)
    G_Walker_mi_Km = G_Walker_mi_Km.to(device)


    dis_Walker_integrate_tensor1 = np.hstack(
        [train_data[4].numpy().T, dis_Walker_integrate_tensor.detach().cpu().numpy()])
    Walker_dis_tensor = torch.FloatTensor(dis_Walker_integrate_tensor1)
    Walker_dis_tensor = Walker_dis_tensor.to(device)

    G_Walker_dis_Kn = ConstructHW.constructHW_knn(Walker_dis_tensor.detach().cpu().numpy(), K_neigs = [13],
                                                  is_probH = False)
    G_Walker_dis_Km = ConstructHW.constructHW_kmean(Walker_dis_tensor.detach().cpu().numpy(), clusters = [9])
    G_Walker_dis_Kn = G_Walker_dis_Kn.to(device)
    G_Walker_dis_Km = G_Walker_dis_Km.to(device)


    mi_Walker_OD_tensor1 = np.hstack(
        [train_data[4].numpy(), mi_Walker_OD_tensor.detach().cpu().numpy()])
    Walker_mi_OD_tensor = torch.FloatTensor(mi_Walker_OD_tensor1)
    Walker_mi_OD_tensor = Walker_mi_OD_tensor.to(device)

    dis_Walker_OD_tensor1 = np.hstack(
        [train_data[4].numpy().T, dis_Walker_OD_tensor.detach().cpu().numpy()])
    Walker_dis_OD_tensor = torch.FloatTensor(dis_Walker_OD_tensor1)
    Walker_dis_OD_tensor = Walker_dis_OD_tensor.to(device)



    TP_X1_list = []  # 存储每轮的X1数据
    TP_Y1_list = []
    TP_X1_knn_list = []
    TP_Y1_knn_list = []
    TP_X1_km_list = []
    TP_Y1_km_list = []

    for epoch in range(1, opt.epoch + 1) :

        score_TP, mi_TP_loss, dis_TP_loss, TP_X1, TP_Y1\
            , TP_mi_feature1, TP_mi_feature2, TP_dis_feature1, TP_dis_feature2, TP_mi_concat_feature, TP_dis_concat_feature\
            = model(Walker_mi_OD_tensor, Walker_dis_OD_tensor,
                                                              G_Walker_mi_Kn, G_Walker_mi_Km, G_Walker_dis_Kn,
                                                              G_Walker_dis_Km)

        TP_X1_np = TP_X1.detach().cpu().numpy()

        TP_X1_list.append(TP_X1_np)

        print(TP_X1_np)

        TP_X1_df = pd.DataFrame(TP_X1_np)

        index1 = pd.DataFrame(train_data[7])
        TP_X1_df = pd.concat([index1, TP_X1_df], axis = 1)
        TP_X1_df.to_csv(f'MDA_DW_mi_全连接降维_epoch_{epoch}.csv', header = None, index = None)

        TP_Y1_np = TP_Y1.detach().cpu().numpy()

        TP_Y1_list.append(TP_Y1_np)

        print(TP_Y1_np)

        TP_Y1_df = pd.DataFrame(TP_Y1_np)
        index2 = pd.DataFrame(train_data[6])
        TP_Y1_df = pd.concat([index2, TP_Y1_df], axis = 1)
        TP_Y1_df.to_csv(f'MDA_DW_dis_全连接降维_epoch_{epoch}.csv', header = None, index = None)

        TP_X1_knn = TP_mi_feature1.detach().cpu().numpy()

        TP_X1_knn_list.append(TP_X1_knn)

        print(TP_X1_knn)

        TP_X1_knn_df = pd.DataFrame(TP_X1_knn)
        index1 = pd.DataFrame(train_data[7])
        TP_X1_knn_df = pd.concat([index1, TP_X1_knn_df], axis = 1)
        TP_X1_knn_df.to_csv(f'MDA_DW_mi_knn_epoch_{epoch}.csv', header = None, index = None)

        TP_Y1_knn = TP_dis_feature1.detach().cpu().numpy()

        TP_Y1_knn_list.append(TP_Y1_knn)

        print(TP_Y1_knn)

        TP_Y1_knn_df = pd.DataFrame(TP_Y1_knn)
        index2 = pd.DataFrame(train_data[6])
        TP_Y1_knn_df = pd.concat([index2, TP_Y1_knn_df], axis = 1)
        TP_Y1_knn_df.to_csv(f'MDA_DW_dis_knn_epoch_{epoch}.csv', header = None, index = None)

        TP_X1_km = TP_mi_feature2.detach().cpu().numpy()

        TP_X1_km_list.append(TP_X1_km)

        print(TP_X1_km)

        TP_X1_km_df = pd.DataFrame(TP_X1_km)
        index1 = pd.DataFrame(train_data[7])
        TP_X1_km_df = pd.concat([index1, TP_X1_km_df], axis = 1)
        TP_X1_km_df.to_csv(f'MDA_DW_mi_km_epoch_{epoch}.csv', header = None, index = None)

        TP_Y1_km = TP_dis_feature2.detach().cpu().numpy()

        TP_Y1_km_list.append(TP_Y1_km)

        print(TP_Y1_km)

        TP_Y1_km_df = pd.DataFrame(TP_Y1_km)
        index2 = pd.DataFrame(train_data[6])
        TP_Y1_km_df = pd.concat([index2, TP_Y1_km_df], axis = 1)
        TP_Y1_km_df.to_csv(f'MDA_DW_dis_km_epoch_{epoch}.csv', header = None, index = None)

        recover_loss_TP = regression_crit(one_index, zero_index, train_data[4].to(device), score_TP)
        reg_loss_TP = get_L2reg(model.parameters())

        tol_loss_TP = recover_loss_TP + mi_TP_loss + dis_TP_loss + 0.00001 * reg_loss_TP
        optim.zero_grad()
        tol_loss_TP.backward()
        optim.step()

    true_value_one_TP, true_value_zero_TP, pre_value_one_TP, pre_value_zero_TP = test(model, train_data, Walker_mi_OD_tensor, Walker_dis_OD_tensor,
                                                              G_Walker_mi_Kn, G_Walker_mi_Km, G_Walker_dis_Kn,
                                                              G_Walker_dis_Km)

    return true_value_one_TP, true_value_zero_TP, pre_value_one_TP, pre_value_zero_TP

def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
    model.eval()
    score,_,_,_,_,_,_,_,_,_,_ = model(concat_mi_tensor, concat_dis_tensor,
                      G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
    test_one_index = data[3][0]
    test_zero_index = data[3][1]
    true_one = data[5][test_one_index]
    true_zero = data[5][test_zero_index]

    pre_one = score[test_one_index]
    pre_zero = score[test_zero_index]

    return true_one, true_zero, pre_one, pre_zero

def train_epoch_MDA(model, train_data, optim, opt):

    model.train()
    regression_crit = Myloss()

    one_index = train_data[2][0]
    zero_index = train_data[2][1]

    dis_sim_integrate_tensor = train_data[0].to(device)
    mi_sim_integrate_tensor = train_data[1].to(device)


    concat_miRNA = np.hstack(
        [train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
    concat_mi_tensor = torch.FloatTensor(concat_miRNA)
    concat_mi_tensor = concat_mi_tensor.to(device)


    G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.detach().cpu().numpy(), K_neigs=[13], is_probH=False)
    G_mi_Km = ConstructHW.constructHW_kmean(concat_mi_tensor.detach().cpu().numpy(), clusters=[9])
    G_mi_Kn = G_mi_Kn.to(device)
    G_mi_Km = G_mi_Km.to(device)


    concat_dis = np.hstack(
        [train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
    concat_dis_tensor = torch.FloatTensor(concat_dis)
    concat_dis_tensor = concat_dis_tensor.to(device)


    G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.detach().cpu().numpy(), K_neigs=[13], is_probH=False)
    G_dis_Km = ConstructHW.constructHW_kmean(concat_dis_tensor.detach().cpu().numpy(), clusters=[9])
    G_dis_Kn = G_dis_Kn.to(device)
    G_dis_Km = G_dis_Km.to(device)


    X1_list = []
    Y1_list = []
    X1_knn_list = []
    Y1_knn_list = []
    X1_km_list = []
    Y1_km_list = []


    for epoch in range(1, opt.epoch+1):

        score, mi_cl_loss, dis_cl_loss, X1, Y1, mi_feature1, mi_feature2, dis_feature1, dis_feature2, mi_concat_feature, dis_concat_feature\
            = model(concat_mi_tensor, concat_dis_tensor,
                    G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km,
                                            )

        X1_np = X1.detach().cpu().numpy()

        X1_list.append(X1_np)

        print(X1_np)

        X1_df = pd.DataFrame(X1_np)

        index1 = pd.DataFrame(train_data[7])
        index1.to_csv(f'MDA_miRNA_序号.csv', header = None, index = None)
        X1_df = pd.concat([index1, X1_df], axis = 1)
        X1_df.to_csv(f'MDA_mi_全连接降维_epoch_{epoch}.csv', header = None, index = None)

        Y1_np = Y1.detach().cpu().numpy()

        Y1_list.append(Y1_np)

        print(Y1_np)

        Y1_df = pd.DataFrame(Y1_np)
        index2 = pd.DataFrame(train_data[6])
        index2.to_csv(f'MDA_disease_序号.csv', header = None, index = None)
        Y1_df = pd.concat([index2, Y1_df], axis = 1)
        Y1_df.to_csv(f'MDA_dis_全连接降维_epoch_{epoch}.csv', header = None, index = None)

        X1_knn = mi_feature1.detach().cpu().numpy()

        X1_knn_list.append(X1_knn)

        print(X1_knn)

        X1_knn_df = pd.DataFrame(X1_knn)
        X1_knn_df = pd.concat([index1, X1_knn_df], axis = 1)
        X1_knn_df.to_csv(f'MDA_mi_knn_epoch_{epoch}.csv', header = None, index = None)

        Y1_knn = dis_feature1.detach().cpu().numpy()

        Y1_knn_list.append(Y1_knn)

        print(Y1_knn)

        Y1_knn_df = pd.DataFrame(Y1_knn)
        Y1_knn_df = pd.concat([index2, Y1_knn_df], axis = 1)
        Y1_knn_df.to_csv(f'MDA_dis_knn_epoch_{epoch}.csv', header = None, index = None)

        X1_km = mi_feature2.detach().cpu().numpy()

        X1_km_list.append(X1_km)

        print(X1_km)

        X1_km_df = pd.DataFrame(X1_km)
        X1_km_df = pd.concat([index1, X1_km_df], axis = 1)
        X1_km_df.to_csv(f'MDA_mi_km_epoch_{epoch}.csv', header = None, index = None)

        Y1_km = dis_feature2.detach().cpu().numpy()

        Y1_km_list.append(Y1_km)

        print(Y1_km)

        Y1_km_df = pd.DataFrame(Y1_km)
        Y1_km_df = pd.concat([index2, Y1_km_df], axis = 1)
        Y1_km_df.to_csv(f'MDA_dis_km_epoch_{epoch}.csv', header = None, index = None)



        recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
        reg_loss = get_L2reg(model.parameters())

        tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00001 * reg_loss
        optim.zero_grad()
        tol_loss.backward()
        optim.step()

    true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(model, train_data, concat_mi_tensor, concat_dis_tensor,
                                 G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)

    return true_value_one, true_value_zero, pre_value_one, pre_value_zero

def main(opt):
    dataset = prepare_data(opt)
    train_data_MDA = Dataset_MDA(opt, dataset)
  
    # for i in range(opt.validation):

    hidden_list = [256, 256]
    num_proj_hidden = 64




    model_MDA = MMLMGNN(467, 72, hidden_list, num_proj_hidden, args)
    model_MDA.to(device)
    optimizer = optim.Adam(model_MDA.parameters(), lr = 0.0001)
    train_data_MDA = Dataset_MDA(opt, dataset)
    true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch_MDA(model_MDA, train_data_MDA[0], optimizer,
                                                                                 opt)
    # DW拓扑结构模态训练
    true_score_one_TP, true_score_zero_TP, pre_score_one_TP, pre_score_zero_TP = train_epoch_MDA_TP(model_MDA, train_data_MDA[0], optimizer,
                                                                                 opt)
    # ST拓扑结构模态训练
    true_score_one_TP, true_score_zero_TP, pre_score_one_TP, pre_score_zero_TP = train_epoch_MDA_ST(model_MDA, train_data_MDA[0], optimizer,
                                                                                 opt)

if __name__ == '__main__':

    args = parameter_parser()
    main(args)

