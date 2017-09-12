# -*- coding:UTF-8 -*-

import random
from numpy import *


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# Node 节点对象计算和记录节点自身的信息(比如输出值、误差项等)，以及与这个节点相关的上下游的连接。
# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
class Node(object):
    def __init__(self, layer_index, node_index):
        '''
        构造节点对象
        :param layer_index:节点所属的层的编号
        :param node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        '''设置节点的输出值，如果节点属于输入层会用到这个函数。'''
        self.output = output

    def append_downstream_connection(self, conn):
        '''添加一个到下游节点的连接'''
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''添加一个到上游节点的连接'''
        self.upstream.append(conn)

    def calc_outppupt(self):
        '''
        根据式1计算节点的输出
        y=sigmoid(wx) (式1)
        '''
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层时，根式式4计算delta
        δ=a(1-a)∑wδ
        :return:
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstreat_node.delta * conn.weight, self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        节点属于输出层时，根据式3计算delta
        δ=y(1-y)(t-y)
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''打印节点的信息'''
        node_str = '%u-%u:output: %f delta: %f' % (self.layer_index, self.node_index,
                                                   self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream' + upstream_str


# ConstNode对象，为了实现一个输出恒为1的节点(计算偏置项时需要)
class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        构造节点对象
        :param layer_index:节点所属的层的编号
        :param node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        '''添加一个到下游节点的连接'''
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层时，根据式4计算delta
        δ=a(1-a)∑wδ
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        '''打印节点的信息'''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


# Layer层对象，由多个节点组成。
# Layer对象，负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作。
class Layer(object):
    def __init__(self, layer_index, node_count):
        '''
        初始化一层
        :param layer_index: 层编号
        :param node_count: 层所包含的节点个数
        '''
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        '''设置层的输出，当层时输入层时会用到。'''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''计算层的输出向量'''
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''打印的信息'''
        for node in self.nodes:
            print node


# Connection每个连接对象都要记录该连接的权重。
# Connection对象，主要职责是记录连接的权重，以及这个连接所关联的上下游节点。
class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        '''
        初始化连接，权重初始化为一个很小的随机数
        :param upstream_node: 连接的上游节点
        :param downstream_node: 连接的下游节点
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        '''计算梯度'''
        self.gradient = self.downstream_node.delta * self.upstream_node.output  # δx

    def get_gradient(self):
        '''获取当前的梯度'''
        return self.gradient

    def update_weight(self, rate):
        '''根据梯度下降算法更新权重'''
        self.calc_gradient()
        self.weight += rate * self.gradient  # w<--w+nδx

    def __str__(self):
        '''打印连接信息'''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


# Connections 仅仅作为Connection的集合对象，提供一些集合操作。
# Connections对象，提供Connection集合操作。
class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connections(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print conn


# Network 神经网络对象，提供API接口。它由若干层对象组成以及连接对象组成。
class Network(object):
    def __init__(self, layers):
        '''
        初始化一个全连接神经网络
        :param layers: 二维数组，描述神经网络每层节点数
        '''
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count-1):
            connections=[Connection(upstream_node,downstream_node)
                         for upstream_node in self.layers[layer].nodes
                         for downstream_node in self.layers[layer+1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connections(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self,labels,data_set,rate,iteration):
        '''
        训练神经网络
        :param labels:数组，训练样本标签。每个元素是一个样本的标签。
        :param data_set:二维数组，训练样本特征。每个元素是一个样本的特征。
        :param rate:
        :param iteration:
        :return:
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train()


    def train_one_sample(self,label,sample,rate):
        '''内部函数，用一个样本训练网络'''
        pass