import logging
from utils.util import load_logger_yaml
import os

class Logger:
    def __init__(self, env:str='env'):
        '''
        :param env: log name, type:str.it should be written in configs/logger.yaml:scene.
        '''
        self.env = env
        self.logger = logging.getLogger()
        # create a handler，used to write log.
        self.logger_yaml = load_logger_yaml()
        self.file_handle = None
        self.console_handle = None
        self._create_logger()
        formatter = logging.Formatter('%(asctime)s %(pathname)s[line:%(lineno)d] %(levelname)s - %(message)s')
        self.file_handle.setLevel(logging.DEBUG)
        self.file_handle.setFormatter(formatter)
        self.console_handle.setFormatter(formatter)
        self.logger.addHandler(self.file_handle)
        self.logger.addHandler(self.console_handle)

    def _create_logger(self):
        if self.env not in self.logger_yaml['scene'].keys():
            raise OSError('No such scene yaml file:{}'.format(self.env))
        log_path = os.path.join(self.logger_yaml['root_path'], self.logger_yaml['scene'][self.env])
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.file_handle = logging.FileHandler(log_path, encoding='utf-8')
        self.console_handle = logging.StreamHandler()







if __name__ == '__main__':
    logger = Logger(env='gg').logger
    logger.debug('logger debug message')
    logger.info('logger info message')
    logger.warning('logger warning message')
    logger.error('logger error message')
    logger.critical('logger critical message')