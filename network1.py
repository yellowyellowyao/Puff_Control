#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/9 10:44
# @Author  : yaoh
# @email   : hyao666@foxmail.com
# @File    : network2.py
# @Software: PyCharm Community Edition

from api_of_data_prepare import read_data_of_label
from api_of_network import *
from sklearn.model_selection import train_test_split
import random
import numpy as np
import time
from functools import partial

time_start = time.time()
tf.reset_default_graph()
# ============================== 提取网路参数和训练数据  ==============================
nn_config = choice_network_config(1)
weight_variable = partial(weight_variable, if_apply_regu=nn_config["if_apply_regu"])

signal, lab, pre_lab = read_data_of_label()

labs = []
for i in lab:
    if i is -1:
        labs.append([1, 0, 0])
    elif i is 0:
        labs.append([0, 1, 0])
    elif i is 1:
        labs.append([0, 0, 1])


signal = np.reshape(signal, [sum(sample_number), Channels_nums, Signal_max_len, 1])
train_x,test_x,train_y,test_y = train_test_split(signal, labs, test_size=0.10,
                                                 random_state=random.randint(0,100), shuffle=True)

num_batch = len(train_x) // batch_size
print('num_batch is {} '.format(num_batch))

x = tf.placeholder(tf.float32, [None, Channels_nums, Signal_max_len,1],name="input")
y_ = tf.placeholder(tf.float32, [None, nn_config["fc_para2"]["label"]])               # 标签

keep_prob_5 = tf.placeholder(tf.float32,name="dropout1")    # 非全连接层dropout数
keep_prob_75 = tf.placeholder(tf.float32,name="dropout2")   # 全连接层dropout数


def cnn_layer():
    # 第一层卷积
    with tf.name_scope('Conv_1'):
        weight1 = weight_variable(nn_config["nn_para1"]["W_k"])
        bias1 = bias_variable([nn_config["nn_para1"]["W_k"][-1]])
        conv1 = prelu(conv2d(x, weight1, nn_config["nn_para1"]["W_s"]) + bias1)
        pool1 = max_pool_2d(conv1, nn_config["nn_para1"]["W_p"])
        drop1 = dropout(pool1, keep_prob_5)

    with tf.name_scope('Conv_2'):
        weight2 = weight_variable(nn_config["nn_para2"]["W_k"])
        bias2 = bias_variable([nn_config["nn_para2"]["W_k"][-1]])
        conv2 = prelu(conv2d(drop1, weight2, nn_config["nn_para2"]["W_s"]) + bias2)
        pool2 = max_pool_2d(conv2, nn_config["nn_para2"]["W_p"])
        drop2 = dropout(pool2, keep_prob_5)

    with tf.name_scope('Conv_3'):
        weight3 = weight_variable(nn_config["nn_para3"]["W_k"])
        bias3 = bias_variable([nn_config["nn_para3"]["W_k"][-1]])
        conv3 = prelu(conv2d(drop2, weight3, nn_config["nn_para3"]["W_s"]) + bias3)
        pool3 = max_pool_2d(conv3, nn_config["nn_para3"]["W_p"])
        drop3 = dropout(pool3, keep_prob_5)

    with tf.name_scope('Conv_4'):
        weight4 = weight_variable(nn_config["nn_para4"]["W_k"])
        bias4 = bias_variable([nn_config["nn_para4"]["W_k"][-1]])
        conv4 = prelu(conv2d(drop3, weight4, nn_config["nn_para4"]["W_s"]) + bias4)
        pool4 = max_pool_2d(conv4, nn_config["nn_para4"]["W_p"])
        drop4 = dropout(pool4, keep_prob_5)

    # 全连接层
    with tf.name_scope('FC_1'):
        to_fc_size = drop4.get_shape().as_list()[-2:]
        drop5_flat = tf.reshape(drop4, [-1, to_fc_size[0]*to_fc_size[1]])
        Wf1 = weight_variable([to_fc_size[0] * nn_config["nn_para4"]["W_k"][-1], nn_config["fc_para1"]["W_1"]])
        bf1 = bias_variable([nn_config["fc_para1"]["W_1"]])
        dense1 = prelu(tf.matmul(drop5_flat, Wf1) + bf1)
        dropf1 = dropout(dense1, keep_prob_75)

    with tf.name_scope('FC_2'):
        Wf2 = weight_variable([nn_config["fc_para1"]["W_1"], nn_config["fc_para2"]["W_1"]])
        bf2 = bias_variable([nn_config["fc_para2"]["W_1"]])
        dense2 = prelu(tf.matmul(dropf1, Wf2) + bf2)
        dropf2 = dropout(dense2, keep_prob_75)

    # 输出层
    with tf.name_scope('out'):
        Wout = weight_variable([nn_config["fc_para2"]["W_1"], nn_config["fc_para2"]["label"]])
        bout = bias_variable([nn_config["fc_para2"]["label"]])
        out = tf.add(tf.matmul(dropf2, Wout), bout, name="out")
    return out


def cnn_train():
    out = cnn_layer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))
    train_step = eval(optimizer)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    saver = tf.train.Saver((tf.global_variables()), max_to_keep=max_to_keep)

    with tf.Session() as sess:
        # 将loss与accuracy保存以供tensorboard使用
        tf.summary.scalar('loss', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)

        # 写入步长
        written_step = tf.Variable(0, name="global_step")
        next_global_step = written_step.assign_add(1)

        sess.run(tf.global_variables_initializer())
        print("CNN layer is {}".format(out))
        # 载入模型再训练
        if if_load_model:
            saver.restore(sess, tf.train.latest_checkpoint(load_model_path))
            # saver.restore.load_variables_from_checkpoint(sess, load_model_path)
            global_step = written_step.eval(sess)
        else:
            global_step = 0

        print("global_step is: ", global_step)
        # 保存图结构
        merged_summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_log_dir, graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(val_log_dir)

        # 早停算法
        best_val, wrong_step = 0, 0
        try:
            while True:
                for i in range(num_batch):
                    global_step += 1
                    batch_x = train_x[i*batch_size : (i+1)*batch_size]
                    batch_y = train_y[i*batch_size : (i+1)*batch_size]

                    _, _, train_acc, loss, train_summary = sess.run(
                        [next_global_step,train_step, accuracy,cross_entropy, merged_summary_op],
                        feed_dict={x: batch_x, y_: batch_y,  keep_prob_5: dropout_of_cnn, keep_prob_75: dropout_of_fc})

                    train_writer.add_summary(train_summary, global_step)
                    # 打印损失
                    # print("step: {} ,acc: {:.4}, loss: {} .  ".format(global_step, train_acc, loss,))

                    if global_step % 100 == 0:

                        # 获取测试数据的准确率
                        val_acc, loss, val_summary = sess.run([accuracy, cross_entropy, merged_summary_op],
                                                              feed_dict={x: test_x, y_: test_y, keep_prob_5: 1.,
                                                              keep_prob_75: 1.})
                        validation_writer.add_summary(val_summary, global_step)

                        acc1 = accuracy.eval({x:batch_x, y_:batch_y, keep_prob_5:dropout_of_cnn, keep_prob_75:dropout_of_fc})
                        loss1 = cross_entropy.eval({x:batch_x, y_:batch_y, keep_prob_5:dropout_of_cnn, keep_prob_75:dropout_of_fc})
                        acc2 = accuracy.eval({x:batch_x, y_:batch_y, keep_prob_5:1.0, keep_prob_75:1.0})
                        loss2 = cross_entropy.eval({x:batch_x, y_:batch_y, keep_prob_5:1.0, keep_prob_75:1.0})

                        print("{:}    valid    accuracy rate is {:<.4},and loss is {:}".format(global_step, val_acc,loss))
                        print("{:}   dropout   accuracy rate is {:<.4},and loss is {:}".format(global_step, acc1,loss1))
                        print("{:}  no-dropout accuracy rate is {:<.4},and loss is {:}\n".format(global_step, acc2, loss2))

                        # 早停算法
                        if val_acc >= best_val:
                            best_val = val_acc
                            wrong_step = 0
                            saver.save(sess, model_dir + '/conv.ckpt', global_step=global_step)
                            step = written_step.eval()
                            print(written_step,step)
                        else:
                            wrong_step += 1
                            if wrong_step > patience:
                                raise Exception("Lost patience !!!")

                    if global_step > max_train_step:
                        raise Exception("Out of  max_train_step !!!")

        except Exception as e:
            print("\n", e, "\n", e, "\n", e, "\n")

        print("Totally cost time is {:.6}s".format(time.time() - time_start))


cnn_train()


