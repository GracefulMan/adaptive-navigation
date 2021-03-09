'''
this class including node class and graph class.
'''

from typing import List, Dict
from utils.util import load_dataset_yaml
from utils.logger import Logger
from env.datasets import DatasetLoader
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import datetime

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

    def __init__(self) -> None:
        self.node_nums = config_data['graph']['node_nums']
        self.img_save_path = config_data['graph']['img_save_path']
        self.graph = self._generate_random_graph(self.node_nums)
        self.image_graph = None
        self.datasets = DatasetLoader()

    def _create_graph(self) -> None:
        pass

    def _generate_random_graph(self,node_nums: int) -> Dict[int, List]:
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
            number = random.randint(1, min(node_nums, 5))  # 随机取1-n个结点
            for i in range(number):
                index = random.randint(0, node_nums - 1)  # 某个结点的位置
                node_append = node_list[index]
                if node_append not in graph[node] and node != node_append:
                    graph[node].append(node_append)
                    graph[node_append].append(node)
        return graph

    def plot_graph(self) -> None:
        '''
        plot the graph and save file to output/graph.
        :return:
        '''
        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)
        G = nx.Graph()
        # use current time to form save path.
        time_now = datetime.datetime.now()
        save_path = os.path.join(self.img_save_path, time_now.strftime("%Y_%m_%d_%H_%M_%S") + ".png")
        for key, values in self.graph.items():
            for edge in values:
                G.add_edge(key, edge)
        nx.draw(G, pos=nx.spring_layout(G), node_color = 'w',edge_color = 'r',with_labels = True, font_size =10,node_size =40,alpha=0.7)
        plt.savefig(save_path)











if __name__ == "__main__":
    graph = Graph()
    graph.plot_graph()











