'''
this class including node class and graph class.
'''

from typing import List, Dict, Any,Set
from utils.util import load_dataset_yaml
from env.datasets import DatasetLoader, logger
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import datetime
import json

config_data = load_dataset_yaml()


class Node:
    '''
    each node contains images for each season.
    '''
    def __init__(self, id:int = 0) -> None:
        self._imgs = {}
        self._id = id

    def load_imgs(self, imgs: dict) -> None:
        self._imgs = copy.copy(imgs)

    def get_season_imgs(self, season: str) -> np.ndarray:
        return self._imgs[season]

    @property
    def id(self) -> int:
        return self._id

    @property
    def imgs(self) -> Dict[str, np.ndarray]:
        return self._imgs


class Graph:
    def __init__(self, preload=True) -> None:
        self._node_nums = config_data['graph']['node_nums']
        self.img_save_path = config_data['graph']['img_save_path']
        self.graph = self._generate_random_graph(self._node_nums) if not preload else self._load_graph()
        self.image_graph = None
        self._datasets = DatasetLoader()
        self.img_graph = {}
        self._create_graph()
        self._current_pos = 0
        self._explored = {}
        self._pos = None

    def get_data_by_index(self, index, season = None) -> Any:
        if season is None:
            return self.img_graph[index].imgs
        else:
            return self.img_graph[index].get_season_imgs(season)

    def _get_edge_nums(self) -> int:
        '''
        get total number of edges.
        :return:
        '''
        tmp = list(self.graph.values())
        res = 0
        for item in tmp:
            res += len(item)
        return res//2

    def _create_graph(self) -> None:
        self._node_nums = min(self._node_nums, self._datasets.img_nums)
        for key in self.graph.keys():
            imgs = self._datasets.get_all_season_for_one_node(key)
            tmpNode = Node(id=key)
            tmpNode.load_imgs(imgs)
            self.img_graph[key] = tmpNode
        logger.info("Construct Graph complete. Total nodes:{}, Total edges:{}".format(self._node_nums, self._get_edge_nums()))

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
        self._pos = nx.spring_layout(G) if self._pos is None else self._pos
        nx.draw(G, pos=self._pos, node_color = 'w',edge_color = 'r',with_labels = True, font_size =10,node_size =40,alpha=0.7)
        plt.savefig(save_path)

    @property
    def current_id(self) -> int:
        return self._current_pos

    @property
    def current_img(self) -> Dict[str, np.ndarray]:
        return self.get_data_by_index(self.current_id)

    @property
    def explored(self) -> Dict[int, Set]:
        return self._explored

    def move(self, id: int) -> None:
        print(self._explored)
        assert id in self.graph[self.current_id], logger.error("current node {} doesn't connect with node {}".format(self.current_id, id))
        # add to explore edge.
        if self._current_pos not in self._explored.keys():
            self._explored[self._current_pos] = set([id])
        else:
            self._explored[self._current_pos].add(id)
        self._current_pos = id

    def get_connected_node(self) -> List:
        return sorted(self.graph[self.current_id])

    def _load_graph(self) -> Dict[int, List]:
        graph_dir = config_data['graph']['predefine_graph_path']
        graph_path = os.path.join(graph_dir, 'graph.json')
        logger.info("Loading graph from path: {}".format(graph_path))
        if not os.path.exists(graph_path):
            os.makedirs(graph_dir)
            graph = self._generate_random_graph(self._node_nums)
            with open(graph_path, 'w') as f:
                json.dump(graph, f)
            return graph
        with open(graph_path, 'r') as f:
            graph = json.load(f)
        # change key type from str to int
        for key in sorted(graph.keys()):
            graph.update({int(key): sorted(graph.pop(key))})
        return graph

    def plot_explore_graph(self):
        #explore graph.
        G = nx.Graph()
        added_edge = {}
        for key, values in self.graph.items():
            for edge in values:
                G.add_edge(key, edge, color='b' if (key in self._explored.keys() and edge in self._explored[key]
                ) else 'r')
        edges, colors = zip(*nx.get_edge_attributes(G, 'color').items())
        self._pos = nx.spring_layout(G) if self._pos is None else self._pos
        nx.draw(G, pos=self._pos, edgelist=edges,edge_color=colors, node_color ='w', with_labels=True, font_size =10,node_size =40,alpha=0.7)
        plt.title('Explored Topological Graph')
        #save figure.
        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)
        # use current time to form save path.
        time_now = datetime.datetime.now()
        save_path = os.path.join(self.img_save_path, time_now.strftime("%Y_%m_%d_%H_%M_%S") + "_explored.png")
        plt.savefig(save_path)















if __name__ == "__main__":
    graph = Graph(preload=True)
    graph.plot_graph()
    id = graph.current_id
    tmp = graph.get_connected_node()
    graph.move(random.choice(tmp))
    tmp = graph.get_connected_node()
    graph.move(random.choice(tmp))
    tmp = graph.get_connected_node()
    graph.move(random.choice(tmp))
    tmp = graph.get_connected_node()
    graph.move(random.choice(tmp))
    tmp = graph.get_connected_node()
    graph.move(random.choice(tmp))
    tmp = graph.get_connected_node()
    graph.move(random.choice(tmp))
    tmp = graph.get_connected_node()
    graph.move(random.choice(tmp))
    tmp = graph.get_connected_node()
    graph.move(random.choice(tmp))
    tmp = graph.get_connected_node()
    graph.move(random.choice(tmp))
    graph.plot_explore_graph()













