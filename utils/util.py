'''
tool func used for env.
'''

import yaml


def load_yaml_file(file_path: str) -> dict:
    '''
    :param file_path: configs file of datasets.
    :return: data of corresponding yaml file.
    '''
    file = open(file_path, 'r', encoding='utf-8')
    data = file.read()
    file.close()
    return yaml.load(data, Loader=yaml.FullLoader)


def load_dataset_yaml() -> dict:
    '''
    load data configs file.
    :return: data. type: dict.
    '''
    path = 'configs/dataset.yaml'
    data = load_yaml_file(path)
    return data


def load_logger_yaml() -> dict:
    '''
    load logger configs file.
    :return: data. type: dict.
    '''
    path= 'configs/logger.yaml'
    data = load_yaml_file(path)
    return data


if __name__ == '__main__':
    load_dataset_yaml()



