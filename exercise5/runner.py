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
    stereo = Stereo(params, K, baseline)

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

    # Disparity movie - Warning! Probably takes a long time to run on your machine
    fig = plt.gcf()
    for i in range(100):
        left_img = Image()
        left_img.load_image(imgs_path + ('/left/%06d.png' % i))
        left_img.imresize(0.5)

        right_img = Image()
        right_img.load_image(imgs_path + ('/right/%06d.png' % i))
        right_img.imresize(0.5)

        Image.imagesc(stereo.get_disparity(left_img.gs, right_img.gs))

        plt.xticks([])
        plt.yticks([])
        plt.show(block=False)
        plt.pause(1)
        plt.clf()

    # Part 3: Create point cloud for first pair
    p_C_points, intensities = stereo.disparity_to_point_cloud(disp_img, left_img.gs)

    # From camera frame to world frame:
    p_F_points = np.matmul(np.linalg.inv(np.array([[0., -1., 0.], [0., 0., -1], [1., 0., 0.]])), p_C_points[0::10, :].transpose()).transpose()

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p_F_points[:,0], p_F_points[:,1], p_F_points[:,2], s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 30])
    ax.set_ylim(params['ylims'])
    ax.set_zlim(params['zlims'])
    plt.show(block=False)
    plt.pause(1)
    plt.clf()

if __name__ == '__main__':

    main()