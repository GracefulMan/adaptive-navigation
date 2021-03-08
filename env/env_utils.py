import cv2
import numpy as np


def show_image(imgs: np.ndarray) -> None:
    '''
    show images
    :param imgs: np.ndarray. shape: (batch_size, height, width, 3) or (height, width, 3)
    :return: None.
    '''
    imgs = np.array(imgs, dtype=np.uint8)
    if len(imgs.shape) == 4:
        tmp = np.empty((imgs.shape[1], 0, imgs.shape[3]))
        for i in range(len(imgs)):
            tmp = np.hstack((tmp, imgs[i].reshape((imgs.shape[1], imgs.shape[2], 3))))
        imgs = tmp[..., ::-1]
        imgs = np.array(imgs, dtype=np.uint8)
        cv2.imshow('', imgs)
        cv2.waitKey()
    else:
        imgs = imgs[..., ::-1]
        cv2.imshow('', imgs)
        cv2.waitKey()