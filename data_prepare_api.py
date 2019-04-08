#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/9 9:21
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : data_prepare_api.py
# @Software: PyCharm Community Edition

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from my_hdf import my_hdf5 as mh
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# ============================== 数据标签准备部分 ==============================
import prepare_data_config as pdc


def check_attr(channels=pdc.Channels,attrs=["T_Freq","T_Start",]):
    """
    检查某个属性
    :param channels:    待检查的通道
    :param attrs:       "T_Start","T_Freq"等属性
    :return: 
    """
    for channel in channels:
        for attr in attrs:
            for shot in range(pdc.start_shot, pdc.end_shot):
                try:
                    print(" {}  {} is ".format(channel,attr),mh.get_attrs(attr, shot_number=shot, channel=channel))
                except AssertionError:
                    # print("  {}  do no have {}".format(shot, channel))
                    continue


def if_all_channel_exist(shot):
    """
    检查是否通道齐全
    :param shot: 
    :return: bool
    """
    for channel in pdc.Channels:
        try:
            mh.if_channel_exist(shot_number=shot, channel=channel)
        except AssertionError:
            return False

    return True


def append_shots_valid(add_shots, isprint=False):
    """
    记录通道齐全的炮号
    :param add_shots: 
    :param isprint: 
    :return: 
    """
    try:
        shots = np.load("./training_database/valid_shots.npy")
    except FileNotFoundError:
        np.save("./training_database/valid_shots.npy", add_shots)
        return
    shots = np.append(shots, add_shots)
    shots = np.array(list(set(shots)))
    shots = np.sort(shots)
    np.save("./training_database/valid_shots.npy",shots)
    if isprint:
        print("Append shots {}: successfully".format(add_shots))


def update_all_label(dir_path="./training_database/",isdel=True):
    """
    合并所有在检查点存储的文件
    :param dir_path: 
    :return: 
    """
    all_label = read_label(dir_path + "all_label.txt")
    for _, _, files in os.walk(dir_path):
        for file in files:
            if file[0:10] == "Checkpoint":    # 如果用切片就不是is
                all_label.update(read_label(dir_path + file))
                if isdel:
                    os.remove(dir_path + file)

    start = min(all_label.keys())
    end = max(all_label.keys())

    file_name = 'all_label.txt'
    with open(dir_path+file_name, 'w') as f:
        f.write(str(all_label))
    print(" {} of {} - {} had been updated successfully".format(file_name, start, end))


def save_label(data, save_path="./training_database/"):
    start = min(data.keys())
    end = max(data.keys())
    with open(save_path + 'Checkpoint of {}-{}.txt'.format(start, end), 'w') as f:
        f.write(str(data))

    print("Saving......  Done!\n")


def read_label(file_path="./training_database/all_label.txt"):
    with open(file_path,'r') as f:
        dict_name = eval(f.read())
    return dict_name


def purify_label(file="./training_database/all_label.txt"):
    """
    统计 all_label 文件中各 label 数量，并删除无效 label
    label 意义参加 prepare_data_config.py
    :param file:  文件位置
    :return:  dict 各 label 数量
    """
    all_label = read_label(file)
    labels_num = {
        -1: 0,
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
    }
    del_shots = []
    for shot, label in all_label.items():
        try:
            labels_num[label] += 1
        except KeyError:
            del_shots.append(shot)

    if len(del_shots) > 0:
        for shot in del_shots:
            del all_label[shot]
        start = min(all_label.keys())
        end = max(all_label.keys())

        with open(file, 'w') as f:
            f.write(str(all_label))
            print(" {} of {} - {} had been purified successfully".format(file, start, end))

    return labels_num


def label_data(shots):
    """
    通过查看各通道信号，给数据打标签，并通过 save_label 存储
    :param shots: 
    :return: 
    """
    data = {}
    for shot in shots:
        if shot in pdc.bad_shots or not if_all_channel_exist(shot):
            continue
        print("shot is: ", shot)

        i = 1
        figure, ax = plt.subplots()
        figure.suptitle('炮号：{} '.format(shot), size=20)
        for channel in pdc.Channels:
            plt.subplot(pdc.fig_height, pdc.fig_width, i)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
            plt.title(channel, size=15)
            plt.xlabel("                                         Time (ms)", size=12)

            Time, value = mh.read_channel(shot_number=shot, channel=channel)
            plt.plot(Time, value)
            if channel is "Ha_MidPlane_1":
                plt.text(200,3,"击穿时间： {:} ms.".format(np.where(value>0.5)[0][0]),size=16)
                plt.plot(40*[10],np.arange(0,4,0.1),linestyle="-.")
            i += 1
        # 打标签
        if pdc.maxshow:
            plt.get_current_fig_manager().window.showMaximized()
        plt.show()

        if not pdc.is_save_label:
            continue

        try:
            data[shot] = int(input("please input label: "))
        except ValueError:

            continue
        if len(data) == pdc.save_numbers:
            save_label(data)
            data = {}

# ============================== 训练数据准备部分 ==============================
import training_config as tc


def ip_subtract_target_ip(shot):
    _, ip = mh.read_channel(shot, channel="IP")
    _, target_ip = mh.read_channel(shot, channel="Target_IP")
    ip_len = min(len(ip),len(target_ip),tc.Channels_len["IP"])
    return ip[0:ip_len]-target_ip[0:ip_len]


def max_of_hx(shot):
    _, hx1 = mh.read_channel(shot, channel="HX_1")
    _, hx2 = mh.read_channel(shot, channel="HX_2")
    # hx_len = min(len(hx1), len(hx2))
    return np.maximum(hx1, hx2)


def re_sampling_ha(shot):
    # 采样率不一样，只能先这样将就了
    _, ha = mh.read_channel(shot, channel="Mag_detla_H")
    return ha[::10]


def padding_len_to_same(shot):
    """
    将数据扩充为统一长度
    :param shot: 炮号 
    :return:     包含5个通道的字典
    """
    _, gas = mh.read_channel(shot, "GasPuffing_FeedForward")
    _, m_d_h = mh.read_channel(shot, "Mag_detla_H")
    data = {
        "IP":                   ip_subtract_target_ip(shot),
        "HX":                   max_of_hx(shot),
        "Ha_MidPlane_1":        re_sampling_ha(shot),
        "Mag_detla_H":          m_d_h,
        "GasPuffing_"
        "FeedForward":          gas[500:],
    }
    del gas, m_d_h
    for channel, value in data.items():
        channel_len = min(tc.Channels_len[channel], len(value))
        data[channel] = np.append(value[0:channel_len], np.zeros(tc.Signal_max_len - channel_len))
    return data


def stack_all_channel(data):
    """
    将 data 的所有 channel 合并为一个矩阵
    :param data:  type:  dict
    :return:      rtype: np.array()
    """
    stack = data[tc.training_channels[0]]
    for signal in tc.training_channels[1:]:
        stack = np.row_stack((stack, data[signal]))
    return stack


def save_data(file_path, save_name):
    """
    制作 all_label 对应的 pickle 文件.(type:dict)
    key: shot , value: channel 
    :param file_path:   从该文件中读取 label
    :param save_name:   存储名，默认 .pkl 文件，后缀会自动补齐
    :return: None
    """
    dir_path = "./training_database/"
    all_label = read_label(dir_path+file_path)
    data_set = {}
    for shot, label in all_label.items():
        if label in [1, 0, -1]:
            data = padding_len_to_same(shot)
            data = stack_all_channel(data)
            data_set[shot] = data

    if not save_name.endswith == ".pkl":
        save_name += ".pkl"

    with open(dir_path + save_name, 'wb') as f:
        pickle.dump(data_set,f)
    print("Saving "+save_name+ " ......  Done!\n")


def read_pickle(file_path):
    with open(file_path,"rb") as f:
        return pickle.load(f)


def gen_pre_lab(shot, all_label, n=10):
    """
    提取 shot 前面 n 炮的 label
    :param shot:        炮号
    :param all_label:   存储 label 的字典
    :param n:           前面 n 炮标签
    :return:            list: 0 or 1
    """
    return [all_label[pre_shot] if pre_shot in all_label.keys() else 0 for pre_shot in range(shot-n, shot)]


def read_data_of_label(sample=tc.sample_number):
    """
    
    :param sample:  list: 不同 label 的数量
    :return: 
    """
    all_data = read_pickle("./training_database/all_data.pkl")
    all_label = read_label("./training_database/all_label.txt")
    data = []
    lab = []
    pre_lab = []

    def append_data(shot):
        data.append(all_data[shot])
        lab.append(all_label[shot])
        pre_lab.append(gen_pre_lab(shot, all_label))

    num__1,num_0, num_1 = 0, 0, 0
    while True:
        for shot in all_label.keys():
            if all_label[shot] is -1 and num__1 < sample[0]:
                append_data(shot)
                num__1 += 1
            elif all_label[shot] is 0 and num_0 < sample[1]:
                append_data(shot)
                num_0 += 1
            elif all_label[shot] is 1 and num_1 < sample[2]:
                append_data(shot)
                num_0 += 1

            if len(lab) >= sum(sample):
                return data, lab, pre_lab


# ============================== 网络操作部分 ==============================
def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:    checkpoint路径(无需加后缀)
    :param output_graph:        PB模型保存路径
    :return:
    '''
    import tensorflow as tf
    # checkpoint = tf.train.get_checkpoint_state(model_folder) # 检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path # 得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # output_node_names = "out/out,input,Placeholder_2,Placeholder_1"
    output_node_names = "out/out,input,dropout1,dropout2"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出