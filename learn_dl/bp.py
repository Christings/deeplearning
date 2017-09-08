# -*- coding:UTF-8 -*-

import random
from numpy import *


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


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
        y=sigmoid(wx) 式1
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

class Layer(object):
    def __init__(self,layer_index,node_count):
        '''
        初始化一层
        :param layer_index: 层编号
        :param node_count: 层所包含的节点个数
        '''
        self.layer_index=layer_index
        self.nodes=[]
        for i in range(node_count):
            self.nodes.append(Node(layer_index,i))
        self.nodes.append(ConstNode(layer_index,node_count))

    def set_output(self,data):
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

# 职责：记录连接的权重，以及这个连接所关联的上下游节点
class Connection(object):
    def __init__(self,upstream_node,downstream_node):
        '''
        初始化连接，权重初始化为一个很小的随机数
        :param upstream_node: 连接的上游节点
        :param downstream_node: 连接的下游节点
        '''
        self.upstream_node=upstream_node
        self.downstream_node=downstream_node
        self.weight=random.uniform(-0.1,0.1)
        self.gradient=0.0

    def calc_gradient(self):
        '''计算梯度'''
        self.gradient=self.downstream_node.delta*self.upstream_node.output

    def get_gradient(self):
        '''获取当前的梯度'''
        return self.gradient
    def update_weight(self,rate):
        '''根据梯度下降算法更新权重'''
        self.calc_gradient()
        self.weight+=rate*self.gradient
    def __str__(self):
        '''打印连接信息'''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)

# 提供Connection集合操作
class Connections(object):
    def __init__(self):
        self.connections=[]
    def add_connections(self,connection):
        self.connections.append(connection)
    def dump(self):
        for conn in self.connections:
            print conn

# 提供API
class Network(object):
    def __init__(self,layers):
        '''
        初始化一个全连接神经网络
        :param layers: 二维数组，描述神经网络每层节点数
        '''
        self.connections=Connections()
        self.layers=[]
        layer_conut=len(layers)
        node_count=0
        for i in range(layers)