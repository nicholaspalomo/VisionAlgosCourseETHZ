from operator import matmul
from matplotlib import image
import numpy as np
import cv2 # OpenCV
import math
from matplotlib import pyplot as plt
import os

from ..utils.process_text_file import ProcessTextFile

class Image:
    def __init__(self):
        self.gs = []

    def load_image(self, image_fname, display=False):
        color_image = cv2.imread(image_fname)
        self.gs = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        if display:
            Image.imshow(self.gs)

        return

    def imresize(self, scale):

        self.gs = cv2.resize(self.gs, dsize=(int(scale * self.gs.shape[1]), int(scale * self.gs.shape[0])), interpolation=cv2.INTER_CUBIC)

        return

    def display_image(self):
        plt.imshow(self.gs)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show(block=False)

        return

    @staticmethod
    def imagesc(img):

        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-5)

        Image.imshow(img_norm)

        return

    @staticmethod
    def normalize(img):

        return (img - img.min()) / (img.max() - img.min() + 1e-5)

    @staticmethod
    def imshow(img):

        plt.imshow(img)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show(block=False)

        return