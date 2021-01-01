# Runner script for exercise 8 (KLT)

import matplotlib as mtplb
import numpy as np
import argparse as ap
import pathlib
from matplotlib import pyplot as plt
import cv2 # OpenCV
import os
import sys
sys.path.append(".")

from tools.utils.image import Image
from tools.algo.klt import KLT

img = Image()
imgs_path = str(pathlib.Path(__file__).parent.absolute()) + "/data"
img.load_image(imgs_path + '/000000.png')

# Part 1: Warping images
# fig, ax = plt.subplots(2, 2)
# ax[0,0].imshow(img.gs)
# ax[0,0].set_title('Reference image')

klt = KLT()
# W = klt.get_sim_warp(50, -30, 0, 1)
# img_warped = klt.warp_image(img.gs, W)
# ax[0,1].imshow(img_warped)
# ax[0,1].set_title('Translation')

# W = klt.get_sim_warp(0, 0, 10, 1)
# img_warped = klt.warp_image(img.gs, W)
# ax[1,0].imshow(img_warped)
# ax[1,0].set_title('Rotation around upper left corner')

# W = klt.get_sim_warp(0, 0, 0, 0.5)
# img_warped = klt.warp_image(img.gs, W)
# ax[1,1].imshow(img_warped)
# ax[1,1].set_title('Zoom on upper left corner')

# plt.show()

# Part 2: Warped patches and recovering a simple warp with brute force
# fig, ax = plt.subplots(1, 2)
# ax[0].get_xaxis().set_visible(False)
# ax[0].get_yaxis().set_visible(False)

# ax[1].get_xaxis().set_visible(False)
# ax[1].get_yaxis().set_visible(False)

# W0 = klt.get_sim_warp(0, 0, 0, 1)
x_T = np.array([[899],[290]])
r_T = 15
# template = klt.get_warped_patch(img.gs, W0, x_T, r_T)

# ax[0].imshow(Image.normalize(template))
# ax[0].set_title('Template')

W = klt.get_sim_warp(10, 6, 0, 1)
img_warped = klt.warp_image(img.gs, W)
r_D = 20
# dx, ssds = klt.track_brute_force(img.gs, img_warped, x_T, r_T, r_D)

# ax[1].imshow(Image.normalize(ssds))
# ax[1].set_title('SSDs')

# plt.show()

# print("Displacement best explained by (dx, dy) = ({}, {})".format(str(dx[0]), str(dx[1])))

# Part 3: Recovering the warp with KLT
num_iters = 50
W, p_hist = klt.track_klt(img.gs, img_warped, x_T, r_T, num_iters)

print("Point moved by {}, should move by (-10, -6)".format(W[:,-1]))

# Part 4: Applying KLT to KITTI
img.gs = cv2.resize(img.gs, dsize=(200, 100), interpolation=cv2.INTER_CUBIC)