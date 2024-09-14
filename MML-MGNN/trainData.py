from __future__ import division


class Dataset_MDA(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index):

        # ID IM为相似性融合矩阵
        # md train和test均为所有样本 正样本和负样本的关系对序号表示
        # md_p md_true为关系的邻接矩阵
        # independent train和test均为所有样本 正样本和负样本的关系对序号表示
        # SD_OD SM_OD为随机游走的欧式距离计算
        # SD_S SM_S为随机游走特征

        return (self.data_set['ID'], self.data_set['IM'],
                self.data_set['md'][index]['train'], self.data_set['md'][index]['test'],
                self.data_set['md_p'], self.data_set['md_true'],
                self.data_set['row_indexes_MDA'], self.data_set['column_indexes_MDA'],

                # self.data_set['independent'][0]['train'],self.data_set['independent'][0]['test'],
                self.data_set['MDA_SD_OD'], self.data_set['MDA_SM_OD'],
                self.data_set['MDA_SD_S'], self.data_set['MDA_SM_S'],

                self.data_set['MDA_SD_OD_ST'], self.data_set['MDA_SM_OD_ST'],
                self.data_set['MDA_SD_S_ST'], self.data_set['MDA_SM_S_ST'],
                )

    def __len__(self):
        return self.nums








