'''
class for load dataset.
'''
from utils.util import load_dataset_yaml
from typing import Dict, List
from glob import glob
from utils.logger import Logger
from env.env_utils import show_image
import numpy as np
import cv2


dataset_configs = load_dataset_yaml()
logger = Logger(env='env').logger


class DatasetLoader:
    '''
    image data loader.
    '''

    def __init__(self) -> None:
        self.data_path = dataset_configs['data_path']['test_data_path']
        # the number of images for each Node.
        self.per_node_image_nums = dataset_configs['node']['image_nums']
        # the interval between two nodes.
        self.sample_interval = dataset_configs['node']['sampling_interval']

        self.img_size = (dataset_configs['image_size']['width'], dataset_configs['image_size']['height'])
        self.imgs_path = {} # all path of images. {'spring':[[1,2,3, ...], [50, 51, ...],...],'summer':[[],[],...]}
        self.imgs = {} # all images. {'spring':[(batch, h, w, 3), (batch, h, w, 3)], 'summer':[]} or list.
        # get all path of images.
        self._get_all_path_for_datasets()
        self._load_image_to_memory()

    def _get_all_path_for_datasets(self) -> None:
        folder_base_name = "/{}_images_test/*/*.png"
        seasons = ['fall', 'spring', 'summer', 'winter']
        for season in seasons:
            path = self.data_path + folder_base_name.format(season)
            imgs = sorted(glob(path))
            self.imgs_path[season] = imgs

    def _load_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=self.img_size)
        return img[..., ::-1]

    def _load_image_to_memory(self, batch_mode: bool = True) -> None:
        img_index = []
        node_num = 0 # calculate node nums.
        for i in range(0, len(self.imgs_path['fall']), self.sample_interval):
            node_num += 1
            tmp = []
            for j in range(i, i+self.per_node_image_nums):
                tmp.append(j)
            img_index.append(tmp)
        logger.info("total node num:{}, total img num:{}".format(node_num, node_num * self.per_node_image_nums))
        logger.info('start to load dataset...')
        for key in self.imgs_path.keys():
            self.imgs[key] = []
            for i in range(len(img_index)):
                if not batch_mode:
                    '''
                    using list to save the img data.
                    '''
                    tmp = []
                    for j in range(len(img_index[i])):
                        # load image.
                        img = self._load_image(self.imgs_path[key][j])
                        tmp.append(img)
                    self.imgs[key].append(tmp)
                else:
                    '''
                    using numpy ndarray to save a batch of images.
                    '''
                    tmp = np.empty(shape=(0, self.img_size[0], self.img_size[1], 3))
                    for j in img_index[i]:
                        img = self._load_image(self.imgs_path[key][j])
                        img = img[np.newaxis, :]
                        tmp = np.vstack((tmp, img))
                self.imgs[key].append(tmp)
        logger.info('dataset loading completed...')

    def get_all_season_for_one_node(self, node: int, visualize: bool=False) -> np.ndarray:
        seasons = ['fall', 'spring', 'summer', 'winter']
        res = np.empty((self.per_node_image_nums, 0, self.img_size[1], 3))
        for season in seasons:
            tmp = self.imgs[season][node]
            res = np.hstack((res, tmp))
        if visualize:
            show_image(res)
        return res






















if __name__ == "__main__":
    dataloader = DatasetLoader()
    dataloader.get_all_season_for_one_node(32,True)
