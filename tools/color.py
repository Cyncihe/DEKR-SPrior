# -*- coding: utf-8 -*-
# @Time    : 2022/6/13 13:48
# @Author  : hjj
# @contact : hejingjing@zhejianglab.com
# @Site    : 
# @File    : color.py
# @Software: PyCharm
import cv2
import numpy as np

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7',
               '009DFF' )#最后一个为灰色
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def hex2rgb(h):  # rgb order (PIL)
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def color_hex(hex):
    # hex = '#' + hex
    return hex2rgb(hex)

colors = Colors()  # create instance for 'from utils.plots import colors'


def draw_pods(image, anns, alpha=0.5):
    """
    draw keypoints and connection lines on the image
    """
    BeanPairsRender = [
        (0, 1), (1, 2), (2, 3), (3, 4)
    ]

    BeanColors = [
        [51, 51, 255], [51, 153, 255], [51, 255, 255], [51, 255, 153], [255, 153, 51]]

    # BeanColors = [
    #     [255, 0, 170], [51, 153, 255], [51, 255, 255], [51, 255, 153], [255, 153, 51]]

    # BeanLineColors = [[255, 153, 153], [255, 153, 204], [255, 255, 204], [204, 255, 229]]
    BeanLineColors = [[255, 153, 153], [0, 170, 255], [255, 255, 204], [204, 255, 229]]
    # BeanLineColors = [[255, 0, 255], [0, 170, 255], [255, 255, 204],  [255, 255, 0],[255, 153, 204], [255, 255, 204], [204, 255, 229]]

    # BeanColors = [
    #     [51, 51, 255], [51, 255, 255], [51, 255, 153], [0, 85, 255], [255, 153, 51]]
    # BeanLineColors = [[51, 51, 255], [51, 255, 255], [51, 255, 153], [0, 85, 255]]

    overlay = image.copy()

    for id, s in enumerate(anns):
        centers = {}
        # s = ann['keypoints']
        nb_keypoints = len(s)
        x = s[:, 0]
        y = s[:, 1]
        z = s[:, 2]

        for i in range(nb_keypoints):
            if z[i] != 0:
                center = np.round(x[i]).astype(int), np.round(y[i]).astype(int)
                cv2.circle(image, center, 3, BeanColors[i], thickness=3, lineType=8, shift=0)
                cv2.circle(overlay, center, 3, BeanColors[i], thickness=3, lineType=8, shift=0)
                centers[i] = center

        # draw line
        for pair_order, pair in enumerate(BeanPairsRender):
            if not z[pair[0]] or not z[pair[1]]:
                continue
            cv2.line(overlay, centers[pair[0]], centers[pair[1]], BeanLineColors[pair_order], 3)
    # Transparency value
    # alpha = 0.6

    # Perform weighted addition of the input image and the overlay

    overlay = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    # overlay = overlay.astype(np.uint8)
    return overlay




def draw_pods_with_aplha(image, anns, alpha=0.5):
    """
    draw keypoints and connection lines on the image
    """
    BeanPairsRender = [
        (0, 1), (1, 2), (2, 3), (3, 4)
    ]

    BeanColors = [
        [51, 255, 153], [51, 255, 255],  [51, 153, 255], [51, 51, 255], [255, 153, 51]]

    # BeanColors = [
    #     [255, 0, 170], [51, 153, 255], [51, 255, 255], [51, 255, 153], [255, 153, 51]]

    # BeanLineColors = [[255, 153, 153], [255, 153, 204], [255, 255, 204], [204, 255, 229]]
    BeanLineColors = [[255, 255, 204],[0, 170, 255], [255, 153, 153],  [204, 255, 229]]
    BeanLineColors = [[255, 255, 204],[0, 180, 255], [255, 153, 153],  [204, 255, 229]]
    # BeanLineColors = [[255, 255, 204],[0, 255, 170], [255, 153, 153],  [204, 255, 229]]
    # BeanLineColors = [[0, 255, 170], [0, 170, 255], [255, 153, 153],  [204, 255, 229]]
    # BeanLineColors = [[255, 0, 255], [0, 170, 255], [255, 255, 204],  [255, 255, 0],[255, 153, 204], [255, 255, 204], [204, 255, 229]]

    # BeanColors = [
    #     [51, 51, 255], [51, 255, 255], [51, 255, 153], [0, 85, 255], [255, 153, 51]]
    # BeanLineColors = [[51, 51, 255], [51, 255, 255], [51, 255, 153], [0, 85, 255]]

    overlay = image.copy()

    for id, s in enumerate(anns):
        centers = {}
        # s = ann['keypoints']
        nb_keypoints = len(s)
        x = s[:, 0]
        y = s[:, 1]
        z = s[:, 2]

        for i in range(nb_keypoints):
            if z[i] > 0:
                center = np.round(x[i]).astype(int), np.round(y[i]).astype(int)
                cv2.circle(image, center, 6, BeanColors[i], thickness=3, lineType=8, shift=0)
                cv2.circle(overlay, center, 3, BeanColors[i], thickness=3, lineType=8, shift=0)
                centers[i] = center

        # draw line
        for pair_order, pair in enumerate(BeanPairsRender):
            if z[pair[0]] < 1e-6 or z[pair[1]] < 1e-6:
                continue
            cv2.line(overlay, centers[pair[0]], centers[pair[1]], BeanLineColors[pair_order], 3)
    # Transparency value
    # alpha = 0.6

    # Perform weighted addition of the input image and the overlay

    overlay = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    # overlay = overlay.astype(np.uint8)
    return overlay