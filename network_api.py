#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 14:27
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : network_api.py
# @Software: PyCharm Community Edition
import tensorflow as tf
from training_config import *


def weight_variable(shape):
    var = tf.Variable(tf.truncated_normal(shape, stddev=0.01))

    if if_apply_regu is "L1":
        tf.contrib.layers.l1_regularize(nn_l1_regularizer)(var)
    elif if_apply_regu is "L2":
        tf.contrib.layers.l2_regularizer(nn_l2_regularizer)(var)

    return var


def bias_variable(shape):
    return tf.Variable(tf.truncated_normal(shape))


def conv2d(x_input, weight, strides):
    # 计算给定4-D input和filter张量的2-D卷积。
    return tf.nn.conv2d(x_input, weight, strides=strides, padding='VALID')


def max_pool(x, k_s):
    return tf.nn.max_pool(x, ksize=k_s[0], strides=k_s[1], padding='SAME')


def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def prelu(x):
    if relu_model is 1:
        return tf.nn.relu(x)
    elif relu_model is 2:
        return tf.nn.leaky_relu(x, alpha=0.1)