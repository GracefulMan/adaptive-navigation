import logging
from utils.util import load_logger_yaml
import os

class Logger:
    def __init__(self, env:str='env'):
        logger = logging.getLogger()
# create a handler，used to write log.
        logger_yaml = load_logger_yaml()
logger_path = logger_yaml['env']['env_log_path']
if not os.path.exists(logger_path):
    os.makedirs(logger_path)

env_loger = logging.FileHandler(os.path.join(logger_path, 'env.log'),encoding='utf-8')
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
env_loger.setLevel(logging.DEBUG)

env_loger.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(env_loger)
logger.addHandler(ch)


logger.debug('logger debug message')
logger.info('logger info message')
logger.warning('logger warning message')
logger.error('logger error message')
logger.critical('logger critical message')