'''
gym style
'''
from env.ConstructGraph import Graph, logger

import numpy as np

class SeasonEnv:
    def __init__(self, preload:bool=True):
        logger.info('-'*30 + "ENV INIT" + "-"*30)
        self._graph = Graph(preload=preload)
        logger.info('_'*30 + "ENV INIT COMPLETE" + "-"* 30)






if __name__ == '__main__':
    env = SeasonEnv()

