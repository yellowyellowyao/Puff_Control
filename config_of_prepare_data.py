#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/9 9:20
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : config_of_prepare_data.py
# @Software: PyCharm Community Edition


# ============================== 通道参数 ==============================
# Channels = ["Target_IP", "IP","GasPuffing_FeedForward", "GasPuffing_FeedBack",
#              "Mag_detla_H", "Mag_detla_V", "HX_1", "HX_2","HMode_Ha","Ha_MidPlane_1"]
#
# T_Freq = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 10000, 1000]         # 单位：Hz
#
# T_Start = [0.003, 0.0,  -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]               # 单位：s


# "Ha_MidPlane_1"信号最好对比着来看
Channels = ["Target_IP", "IP", "GasPuffing_FeedForward",
            "Ha_MidPlane_1", "HX_1", "HX_2","Mag_detla_H"]


# ============================== 炮号区间 ==============================
# start_shot, end_shot = 34521, 36948  # 预期所有区间

start_shot = 34000
end_shot = 34556

# 预送气有问题的炮号
import intervals
bad_shots = intervals.closed(34556, 34642) | intervals.closed(34771, 34800) |\
            intervals.closed(35900, 35953) | intervals.closed(36396, 36408) |\
            intervals.closed(36763, 36768)
# (34556,34642), (34771, 34800), (35900, 35953), (36396,36408), (36763, 36768)

sequence_mode = 2  # 炮号顺序
# 0: 正常顺序，   1: 乱序，   2: 倒序

# ============================== label与储存 ==============================

# -1：放气少，   0：差不多，  1：放气多，
# 暂时不用的炮类型
# 3: 注气坏了
# 4: Ha坏了
# 5: HX坏了
# 6：放炮时间太短，不用       7：SMBI
# 8：NBI   9：拿不准暂时不用

# is_save_label = False
is_save_label = True
save_numbers = 10      # data达到该数目数据后保存


# =============================== 作图参数 ===============================
# subplot 的 宽与高，高已在程序中体现，无需设置
import math
fig_width = 2
fig_height = math.ceil(len(Channels) / fig_width)

# 是否最大化显示
maxshow = True
