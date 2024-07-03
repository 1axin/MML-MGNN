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

class Dataset_CDA(object) :
    def __init__(self, opt, dataset) :
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index) :
        # ID IM为相似性融合矩阵
        # md train和test均为所有样本 正样本和负样本的关系对序号表示
        # md_p md_true为关系的邻接矩阵
        # independent train和test均为所有样本 正样本和负样本的关系对序号表示
        # SD_OD SM_OD为随机游走的欧式距离计算
        # SD_S SM_S为随机游走特征

        return (
                self.data_set['CDA_ID'], self.data_set['CDA_IM'],
                self.data_set['cd'][index]['train'], self.data_set['cd'][index]['test'],
                self.data_set['cd_p'], self.data_set['cd_true'],

                self.data_set['row_indexes_CDA'], self.data_set['column_indexes_CDA'],
                # self.data_set['independent'][0]['train'], self.data_set['independent'][0]['test'],
                self.data_set['CDA_SD_OD'], self.data_set['CDA_SM_OD'],
                self.data_set['CDA_SD_S'], self.data_set['CDA_SM_S'],

                self.data_set['CDA_SD_OD_ST'], self.data_set['CDA_SM_OD_ST'],
                self.data_set['CDA_SD_S_ST'], self.data_set['CDA_SM_S_ST'],
                )

    def __len__(self) :
        return self.nums

class Dataset_LDA(object) :
    def __init__(self, opt, dataset) :
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index) :
        # ID IM为相似性融合矩阵
        # md train和test均为所有样本 正样本和负样本的关系对序号表示
        # md_p md_true为关系的邻接矩阵
        # independent train和test均为所有样本 正样本和负样本的关系对序号表示
        # SD_OD SM_OD为随机游走的欧式距离计算
        # SD_S SM_S为随机游走特征

        return (
                self.data_set['LDA_ID'], self.data_set['LDA_IM'],
                self.data_set['ld'][index]['train'], self.data_set['ld'][index]['test'],
                self.data_set['ld_p'], self.data_set['ld_true'],

                self.data_set['row_indexes_LDA'], self.data_set['column_indexes_LDA'],
                # self.data_set['independent'][0]['train'], self.data_set['independent'][0]['test'],
                self.data_set['LDA_SD_OD'], self.data_set['LDA_SM_OD'],
                self.data_set['LDA_SD_S'], self.data_set['LDA_SM_S'],

                self.data_set['LDA_SD_OD_ST'], self.data_set['LDA_SM_OD_ST'],
                self.data_set['LDA_SD_S_ST'], self.data_set['LDA_SM_S_ST'],
                )

    def __len__(self) :
        return self.nums

class Dataset_MGA(object) :
    def __init__(self, opt, dataset) :
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index) :
        # ID IM为相似性融合矩阵
        # md train和test均为所有样本 正样本和负样本的关系对序号表示
        # md_p md_true为关系的邻接矩阵
        # independent train和test均为所有样本 正样本和负样本的关系对序号表示
        # SD_OD SM_OD为随机游走的欧式距离计算
        # SD_S SM_S为随机游走特征

        return (
                self.data_set['MGA_ID'], self.data_set['MGA_IM'],
                self.data_set['mg'][index]['train'], self.data_set['mg'][index]['test'],
                self.data_set['mg_p'], self.data_set['mg_true'],

                self.data_set['row_indexes_MGA'], self.data_set['column_indexes_MGA'],
                # self.data_set['independent'][0]['train'], self.data_set['independent'][0]['test'],
                self.data_set['MGA_SD_OD'], self.data_set['MGA_SM_OD'],
                self.data_set['MGA_SD_S'], self.data_set['MGA_SM_S'],

                self.data_set['MGA_SD_OD_ST'], self.data_set['MGA_SM_OD_ST'],
                self.data_set['MGA_SD_S_ST'], self.data_set['MGA_SM_S_ST'],
                )

    def __len__(self) :
        return self.nums

class Dataset_CMI(object) :
    def __init__(self, opt, dataset) :
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index) :
        # ID IM为相似性融合矩阵
        # md train和test均为所有样本 正样本和负样本的关系对序号表示
        # md_p md_true为关系的邻接矩阵
        # independent train和test均为所有样本 正样本和负样本的关系对序号表示
        # SD_OD SM_OD为随机游走的欧式距离计算
        # SD_S SM_S为随机游走特征

        return (
            self.data_set['CMI_ID'], self.data_set['CMI_IM'],
            self.data_set['cm'][index]['train'], self.data_set['cm'][index]['test'],
            self.data_set['cm_p'], self.data_set['cm_true'],

            self.data_set['row_indexes_CMI'], self.data_set['column_indexes_CMI'],
            # self.data_set['independent'][0]['train'], self.data_set['independent'][0]['test'],
            self.data_set['CMI_SD_OD'], self.data_set['CMI_SM_OD'],
            self.data_set['CMI_SD_S'], self.data_set['CMI_SM_S'],

            self.data_set['CMI_SD_OD_ST'], self.data_set['CMI_SM_OD_ST'],
            self.data_set['CMI_SD_S_ST'], self.data_set['CMI_SM_S_ST'],
        )

    def __len__(self) :
        return self.nums

class Dataset_LMI(object) :
    def __init__(self, opt, dataset) :
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index) :
        # ID IM为相似性融合矩阵
        # md train和test均为所有样本 正样本和负样本的关系对序号表示
        # md_p md_true为关系的邻接矩阵
        # independent train和test均为所有样本 正样本和负样本的关系对序号表示
        # SD_OD SM_OD为随机游走的欧式距离计算
        # SD_S SM_S为随机游走特征

        return (
            self.data_set['LMI_ID'], self.data_set['LMI_IM'],
            self.data_set['lm'][index]['train'], self.data_set['lm'][index]['test'],
            self.data_set['lm_p'], self.data_set['lm_true'],

            self.data_set['row_indexes_LMI'], self.data_set['column_indexes_LMI'],
            # self.data_set['independent'][0]['train'], self.data_set['independent'][0]['test'],
            self.data_set['LMI_SD_OD'], self.data_set['LMI_SM_OD'],
            self.data_set['LMI_SD_S'], self.data_set['LMI_SM_S'],

            self.data_set['LMI_SD_OD_ST'], self.data_set['LMI_SM_OD_ST'],
            self.data_set['LMI_SD_S_ST'], self.data_set['LMI_SM_S_ST'],
        )

    def __len__(self) :
        return self.nums






