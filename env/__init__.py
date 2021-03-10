'''
gym style
'''
from env.ConstructGraph import Graph, logger
from typing import Dict, Tuple,List
import numpy as np

class SeasonEnv:
    def __init__(self, preload:bool=True):
        logger.info('-'*30 + "ENV INIT" + "-"*30)
        self._graph = Graph(preload=preload)
        logger.info('_'*30 + "ENV INIT COMPLETE" + "-"* 30)
        self._terminal = False
        self.reset()

    def reset(self) -> Tuple[Dict[str, np.ndarray], List]:
        self._graph.reset()
        self._obesrvation = self._graph.current_img
        self._action_space = self._graph.get_connected_node()
        return self._obesrvation, self._action_space

    def step(self, action:int) -> Tuple[Tuple[Dict[str, np.ndarray], List], float, bool, str]:
        '''
        //TODO: terminal state.
        how to judge terminal state.
        :param action: choose next move id.
        :return: observation, reward, terminal, info
        '''
        if action not in self._action_space:
            return (self.observation, self._action_space), self.reward, self.terminal, 'wrong action'
        self._graph.move(action)
        self._obesrvation = self._graph.current_img
        self._action_space = self._graph.get_connected_node()
        return (self._obesrvation, self._action_space), self.reward, self.terminal, ''

    @property
    def reward(self) -> float:
        return 1. if self.terminal else -0.1

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def observation(self):
        return self._obesrvation

    @property
    def action_space(self):
        return self._action_space

    def explore_graph(self) -> None:
        self._graph.plot_explore_graph()







if __name__ == '__main__':
    env = SeasonEnv()
    obs, actions = env.reset()
    n = 10
    import random

    while n:
        action = random.choice(actions)
        (obs_, actions_), reward, terminal, _ = env.step(action)
        obs = obs_
        actions = actions_
        n -= 1

    env.explore_graph()


