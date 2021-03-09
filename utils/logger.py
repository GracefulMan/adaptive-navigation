import logging
from utils.util import load_logger_yaml
import os
class Logger:
    def __init__(self, env:str='env') -> None :
        '''
        :param env: log name, type:str.it should be written in configs/logger.yaml:scene.
        '''
        self.env = env
        self.logger = logging.getLogger()
        logging.basicConfig(level=logging.DEBUG, filemode='w', filename='/dev/null')
        # create a handler，used to write log.
        self.logger_yaml = load_logger_yaml()
        self.file_handle = None
        self.console_handle = None
        self._create_logger()
        self._get_log_level()
        formatter = logging.Formatter('[{}] %(asctime)s %(pathname)s[line:%(lineno)d] %(levelname)s - %(message)s'.format(self.env))
        self.file_handle.setLevel(self.logger_level)
        self.console_handle.setLevel(self.logger_level)
        self.file_handle.setFormatter(formatter)
        self.console_handle.setFormatter(formatter)
        self.logger.addHandler(self.file_handle)
        self.logger.addHandler(self.console_handle)

    def _get_log_level(self):
        '''
        get log level from yaml file.
        :return:
        '''
        level_dict = {
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "ERROR": logging.ERROR,
            "DEBUG": logging.DEBUG
        }

        level_key = self.logger_yaml['log_level']
        self.logger_level = level_dict[level_key]




    def _create_logger(self) -> None:
        if self.env not in self.logger_yaml['scene'].keys():
            raise OSError('No such scene yaml file:{}'.format(self.env))
        log_path = os.path.join(self.logger_yaml['root_path'], self.logger_yaml['scene'][self.env])
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.file_handle = logging.FileHandler(log_path, encoding='utf-8')
        self.console_handle = logging.StreamHandler()








if __name__ == '__main__':
    logger = Logger(env='env').logger
    logger.debug('logger debug message')
    logger.info('logger info message')
    logger.warning('logger warning message')
    logger.error('logger error message')
    logger.critical('logger critical message')