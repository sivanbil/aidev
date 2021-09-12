# !/usr/bin/env python
# -*-coding:utf-8 -*-
import numpy as np
import scipy
import matplotlib.pyplot

class NeualNetwork():
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置输入层节点，隐藏层节点和输出层节点的数量
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 学习率设置
        self.lr = learningrate

        # 权重矩阵设置，正态分布
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数设置，sigmod函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, input_list, target_list):
        inuputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # 计算到隐藏层的信号
        hidden_inputs = np.dot(self.wih, inuputs)

        # 计算隐藏输出的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算到输出层的信号
        final_inputs = np.dot(self.who, hidden_inputs)

        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors*final_outputs*(0-final_outputs)), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs * (0 - hidden_errors)), np.transpose(inuputs))


        pass

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        print('n')
