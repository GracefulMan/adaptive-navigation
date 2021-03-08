'''
this class including node class and graph class.
'''

from typing import List
from utils.util import load_dataset_yaml
from utils.logger import Logger
from env.datasets import DatasetLoader
logger = Logger(env='env').logger
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import networks as nx


config_data = load_dataset_yaml()

class Node:
    '''
    each node contains images for each season.
    '''
    def __init__(self, id:int = 0) -> None:
        self.imgs = {}
        self.id = id

    def load_imgs(self, imgs: dict) -> None:
        self.imgs = copy.copy(imgs)

    def get_season_imgs(self, season: str) -> np.ndarray:
        return self.imgs[season]


class Graph:

    def __init__(self):
        graph = self.generate_random_graph(4)
        print(graph)

    def generate_random_graph(self,node_nums):
        """
        Generate random graph with number of n
        :param n: graph's number of nodes
        :return: a dict of graph
        """
        number = node_nums
        node_list = []
        node_name = list(range(node_nums))
        graph = {}
        for node in node_name:  # 循环创建结点
            node_list.append(node)
            number -= 1
            if number == 0:
                break
        # print(node_list)

        for node in node_list:
            graph[node] = []  # graph为dict结构，初始化为空列表

        for node in node_list:  # 创建连接无向图（无加权值）
            number = random.randint(1, node_nums)  # 随机取1-n个结点
            for i in range(number):
                index = random.randint(0, node_nums - 1)  # 某个结点的位置
                node_append = node_list[index]
                if node_append not in graph[node] and node != node_append:
                    graph[node].append(node_append)
                    graph[node_append].append(node)
        return graph








if __name__ == "__main__":
    graph = Graph()











