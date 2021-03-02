'''
class for load dataset.
'''
from env.utils import load_dataset_yaml

dataset_configs = load_dataset_yaml()


class DatasetLoader:
    '''
    image data loader.
    '''
    def __init__(self):
        self.train_data_path = dataset_configs['DATA_PATH']['TRAIN_DATA_PATH']
        self.test_data_path = dataset_configs['DATA_PATH']['TEST_DATA_PATH']