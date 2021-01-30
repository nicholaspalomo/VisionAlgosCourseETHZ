# Runner script for exercise 5 (Multiple-view Geometry, Lecture 7)

from cv2 import data
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import argparse as ap
import pathlib
import cv2
import os
import sys
import time

sys.path.append(".")

from tools.utils.image import Image
from tools.utils.process_text_file import ProcessTextFile
from tools.algo.stereo import Stereo

def main():

    imgs_path = str(pathlib.Path(__file__).parent.absolute()) + "/data"
    
    left_img = Image()
    left_img.load_image(imgs_path + '/left/000000.png')
    left_img.imresize(0.5)

    right_img = Image()
    right_img.load_image(imgs_path + '/right/000000.png')
    right_img.imresize(0.5)

    file = ProcessTextFile()
    K = file.read_file(imgs_path + '/K.txt')
    K[:2, :] /= 2

    poses = file.read_file(imgs_path + '/poses.txt')

    # Given by the KITTI dataset
    baseline = 0.54

    # Algorithm parameters
    params = dict(\
        {'patch_radius' : 5,
        'min_disp' : 5,
        'max_disp' : 50,
        'xlims' : [7, 20],
        'ylims' : [-6, 10],
        'zlims' : [-5, 5]
        })

    # Parts 1, 2, and 4: Disparity on one image pair
    stereo = Stereo(params)

    tic = time.time()
    disp_img = stereo.get_disparity(left_img.gs, right_img.gs)
    toc = time.time() - tic
    print("time elapsed: {}".format(toc))

    fig = plt.gcf()
    Image.imagesc(disp_img)
    fig.gca().axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.show(block=False)
    plt.pause(1)
    plt.close()

if __name__ == '__main__':

    main()