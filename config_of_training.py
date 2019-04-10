#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/9 10:49
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : config_of_training.py
# @Software: PyCharm Community Edition


training_channels = ["Mag_detla_H","IP", "Ha_MidPlane_1","GasPuffing_FeedForward","HX",]

# 各通道使用的数据长度 (单位：ms)
Channels_len = {"IP": 200,
                "Ha_MidPlane_1": 200,
                "Mag_detla_H": 200,
                "HX": 500,
                "GasPuffing_FeedForward": 500,
                }


Channels_nums = len(Channels_len)
Signal_max_len = max(Channels_len.values())

from datetime import datetime
time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = r"./model/tensorboard/train/"+time_stamp
val_log_dir = r"./model/tensorboard/validation/"+time_stamp

model_dir = r"./model/checkpoint"

# 用于训练的炮，每次训练抓取样本数，最大训练步数
sample_number = [600, 300, 300]
batch_size = 128
max_train_step = 5000
max_to_keep = 20        # 最大保存 ckpt 文件数

# 梯度下降选择：SGD、Adam
# optimizer = "tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)"
optimizer = "tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)"

# 卷积层与全连接层 dropout 值
dropout_of_cnn = 0.5
dropout_of_fc = 0.75


# 定义L1、L2正则化参数
nn_l1_regularizer = 0.1
nn_l2_regularizer = 0.03

relu_model = 1
# 1: relu
# 2: leaky_relu

# 早停算法
patience = 5

# 是否需要载入模型再训练(如需要，请划分训练、验证数据集的pickle文件)
if_load_model = False
load_model_path = "./model/checkpoint/"


def choice_network_config(network_plan):
    """
    :param network_plan:  int
        1: 共享权值方案 (最佳参数)
        2: 共享权值方案
        3：单个通道单个网络，最后合并 方案
        
    :return:  dict
    """
    if network_plan is 1:
        if_apply_regu = "L2"
        channel_of_each_layers = {
            "cnn1": 32,
            "cnn2": 64,
            "cnn3": 64,
            "cnn4": 64,
            "fc1": 512,
            "fc2": 128,
        }

        # 卷积核strides步长，对于图片，因为只有两维，通常strides取[1，stride，stride，1]
        w_s = [1, 1, 1, 1]
        # 池化层大小及步长
        w_p = [[1, 2, 1, 1], [1, 1, 1, 1]]
        w_size = 50

        nn_para1 = {
            # 卷积核kernel大小(Channels_nums, 30)， 输入通道数(1)， 输出通道数(32)
            "W_k": [1, w_size, 1, channel_of_each_layers["cnn1"]],
            "W_s": w_s,
            "W_p": w_p,
        }

        nn_para2 = {"W_k": [1, w_size, channel_of_each_layers["cnn1"], channel_of_each_layers["cnn2"]],
                    "W_s": w_s,
                    "W_p": w_p, }

        nn_para3 = {"W_k": [1, w_size, channel_of_each_layers["cnn2"], channel_of_each_layers["cnn3"]],
                    "W_s": w_s,
                    "W_p": w_p, }

        nn_para4 = {"W_k": [Channels_nums, w_size, channel_of_each_layers["cnn3"], channel_of_each_layers["cnn4"]],
                    "W_s": w_s,
                    "W_p": w_p, }

        fc_para1 = {"W_1": channel_of_each_layers["fc1"],
                    # 将pre_label添加到此层
                    # "p_l": 10,
                    }

        fc_para2 = {"W_1": channel_of_each_layers["fc2"],
                    # "label": 1,
                    "label": 3,
                    }
        nn_config = {
            "if_apply_regu":        if_apply_regu,
            "channel_of_each_layers": channel_of_each_layers,
            "nn_para1": nn_para1,
            "nn_para2": nn_para2,
            "nn_para3": nn_para3,
            "nn_para4": nn_para4,
            "fc_para1": fc_para1,
            "fc_para2": fc_para2,
        }

    elif network_plan is 2:
        if_apply_regu = "L2"
        channel_of_each_layers = {
            "cnn1": 32,
            "cnn2": 64,
            "cnn3": 64,
            "cnn4": 64,
            "cnn5": 128,
            "cnn6": 128,
            "fc1": 512,
            "fc2": 128,
        }

        # 卷积核strides步长，对于图片，因为只有两维，通常strides取[1，stride，stride，1]
        w_s = [1, 1, 1, 1]
        # 池化层大小及步长
        w_p = [[1, 2, 1, 1], [1, 1, 1, 1]]
        w_size = 30

        nn_para1 = {
            # 卷积核kernel大小(Channels_nums, 30)， 输入通道数(1)， 输出通道数(32)
            "W_k": [1, w_size, 1, channel_of_each_layers["cnn1"]],
            "W_s": w_s,
            "W_p": w_p,
        }

        nn_para2 = {"W_k": [1, w_size, channel_of_each_layers["cnn1"], channel_of_each_layers["cnn2"]],
                    "W_s": w_s,
                    "W_p": w_p, }

        nn_para3 = {"W_k": [1, w_size, channel_of_each_layers["cnn2"], channel_of_each_layers["cnn3"]],
                    "W_s": w_s,
                    "W_p": w_p, }

        nn_para4 = {"W_k": [1, w_size, channel_of_each_layers["cnn3"], channel_of_each_layers["cnn4"]],
                    "W_s": w_s,
                    "W_p": w_p, }

        nn_para5 = {"W_k": [1, w_size, channel_of_each_layers["cnn4"], channel_of_each_layers["cnn5"]],
                    "W_s": w_s,
                    "W_p": w_p, }

        nn_para6 = {"W_k": [Channels_nums, w_size, channel_of_each_layers["cnn5"], channel_of_each_layers["cnn6"]],
                    "W_s": w_s,
                    "W_p": w_p, }

        fc_para1 = {"W_1": channel_of_each_layers["fc1"],
                    # 将pre_label添加到此层
                    # "p_l": 10,
                    }

        fc_para2 = {"W_1": channel_of_each_layers["fc2"],
                    # "label": 1,
                    "label": 3,
                    }

        nn_config = {
            "if_apply_regu": if_apply_regu,
            "channel_of_each_layers": channel_of_each_layers,
            "nn_para1": nn_para1,
            "nn_para2": nn_para2,
            "nn_para3": nn_para3,
            "nn_para4": nn_para4,
            "nn_para5": nn_para5,
            "nn_para6": nn_para6,
            "fc_para1": fc_para1,
            "fc_para2": fc_para2,
        }

    elif network_plan is 3:
        if_apply_regu = False
        channel_of_each_layers = {
            # 一维卷积
            "cnn1": 8,
            "cnn2": 16,
            "cnn3": 16,

            # 二维卷积
            "cnn4": 128,
            "fc1": 512,
            "fc2": 128,
        }

        # 卷积核strides步长，对于图片，因为只有两维，通常strides取[1，stride，stride，1]
        w_s_1d = 1
        w_s_2d = [1, 1, 1, 1]

        # 池化层大小及步长
        w_p_1d = [[2], [1]]
        w_p_2d = [[1, 2, 1, 1], [1, 1, 1, 1]]
        w_size = 50

        nn_para1 = {
            # 卷积核kernel大小(Channels_nums, 30)， 输入通道数(1)， 输出通道数(32)
            "W_k": [w_size, 1, channel_of_each_layers["cnn1"]],
            "W_s": w_s_1d,
            "W_p": w_p_1d,
        }

        nn_para2 = {"W_k": [w_size, channel_of_each_layers["cnn1"], channel_of_each_layers["cnn2"]],
                    "W_s": w_s_1d,
                    "W_p": w_p_1d, }

        nn_para3 = {"W_k": [w_size, channel_of_each_layers["cnn2"], channel_of_each_layers["cnn3"]],
                    "W_s": w_s_1d,
                    "W_p": w_p_1d, }

        nn_para4 = {"W_k": [Channels_nums, w_size, channel_of_each_layers["cnn3"], channel_of_each_layers["cnn4"]],
                    "W_s": w_s_2d,
                    "W_p": w_p_2d, }

        fc_para1 = {"W_1": channel_of_each_layers["fc1"],
                    # 将pre_label添加到此层
                    # "p_l": 10,
                    }

        fc_para2 = {"W_1": channel_of_each_layers["fc2"],
                    # "label": 1,
                    "label": 3,
                    }

        nn_config = {
            "if_apply_regu": if_apply_regu,
            "channel_of_each_layers": channel_of_each_layers,
            "nn_para1": nn_para1,
            "nn_para2": nn_para2,
            "nn_para3": nn_para3,
            "nn_para4": nn_para4,
            "fc_para1": fc_para1,
            "fc_para2": fc_para2,
        }

    return nn_config