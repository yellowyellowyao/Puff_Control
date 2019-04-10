#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 14:46
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : result_present.py
# @Software: PyCharm Community Edition
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from api_of_data_prepare import *
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号


def init_NN(pb_file_path=r"./model/frozen_model.pb"):
    """
    从 pb_file_path 读取pb文件，并返回sess及输入、输出节点
    :param pb_file_path: 
    :return: 
    """
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    finger_input = sess.graph.get_tensor_by_name("input:0")
    output = sess.graph.get_tensor_by_name("out/out:0")
    dropout2 = sess.graph.get_tensor_by_name("dropout2:0")
    dropout1 = sess.graph.get_tensor_by_name("dropout1:0")
    return sess, finger_input, output, dropout1, dropout2
    # return sess, finger_input, output


def padding_len_to_same(shot):
    """
    将数据扩充为统一长度
    :param shot: 炮号 
    :return:     包含5个通道的字典
    """
    _, gas = mh.read_channel(shot, "GasPuffing_FeedForward")
    _, m_h = mh.read_channel(shot, "Mag_detla_H")
    data = {
        "IP":                   ip_subtract_target_ip(shot),
        "HX":                   max_of_hx(shot),
        "Ha_MidPlane_1":        re_sampling_ha(shot),
        "Mag_detla_H":          m_h,
        "GasPuffing_"
        "FeedForward":          gas[500:],
    }
    del gas, m_h
    for channel, value in data.items():
        channel_len = min(tc.Channels_len[channel], len(value))
        data[channel] = np.append(value[0:channel_len], np.zeros(tc.Signal_max_len - channel_len))
    return data


def my_softmax(logits):
    logits = [pow(1.1, x) for x in logits]
    proportion = logits / sum(logits)
    return proportion, -(proportion[0]*-7 + proportion[-1]*4)/(pow(0.85+proportion[1],1.15))


def result_present(shots=list(range(34000, 36948))):
    # =============================加载pb模型=================================
    # sess, finger_input, output, Placeholder_2,Placeholder_1 = init_NN()
    sess, finger_input, output, dropout1, dropout2 = init_NN(r"./model/frozen_model.pb")
    random.shuffle(shots)
    for shot in shots:
        if shot in pdc.bad_shots:
            continue

        # 检查通道是否齐全
        if not if_all_channel_exist(shot):
            continue
        print("shot: ",shot)
        # =============================画出各个通道=================================
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
                try:
                    plt.text(200,3,"击穿时间： {:} ms.".format(np.where(value>0.5)[0][0]),size=16)
                except IndexError:
                    pass
            elif channel is "GasPuffing_FeedForward":
                try:
                    gas_puf = np.where(value[0:1000]>50)[0]
                    gas_puf = gas_puf[-1] - gas_puf[0]
                except IndexError:
                    gas_puf = 0

            i += 1

        plt.subplot(pdc.fig_height, pdc.fig_width, i)
        data = padding_len_to_same(shot)
        stack = data[tc.training_channels[0]]
        for signal in tc.training_channels[1:]:
            stack = np.row_stack((stack, data[signal]))

        stack = np.reshape(stack, [1,5, 500, 1])
        logits = sess.run(output, feed_dict={finger_input: stack, dropout1:1.0, dropout2:1.0})
        # predict = sess.run(output, feed_dict={finger_input: stack,Placeholder_2:1.0,Placeholder_1:1.0 })

        # =============================自定义softmax=================================
        logits = logits[0]
        plt.text(0, 1.2, "       少       正好    多 ", size=10)
        plt.text(0, 0.9, "logits:{} ".format(logits), size=10)
        proportion,songqi = my_softmax(logits)

        plt.text(0, 0.7, "propor:{} ".format(proportion), size=10)
        # print("推荐送气量： {:.2} ms".format(songqi))

        plt.text(0.1, 0.25, "当前送气量:{:} ms \n推荐送气量:{:.2} ms".format(gas_puf,songqi), size=10)
        if songqi > 2:
            plt.text(0.5, 0.25, "推荐增加送气量:{:.2} ms".format(songqi),size=16)
        elif songqi < -2:
            plt.text(0.5, 0.25, "推荐减少送气量:{:.2} ms".format(-songqi),size=16)
        else:
            plt.text(0.5, 0.25, "推荐维持送气量",size=16)

        plt.get_current_fig_manager().window.showMaximized()
        plt.show()

# shots=list(range(33000, 34500))
# shots = [36867]

# 只查看标签labels的炮
shots = []
labels = [1,0,-1]
all_label = read_label()
for shot,label in all_label.items():
    if label in labels:
        shots.append(shot)

print("shot number:", len(shots))

with tf.device('/cpu:0'):
    result_present(shots)

# result_present()
