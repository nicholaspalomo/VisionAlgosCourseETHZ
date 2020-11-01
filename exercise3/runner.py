from os import device_encoding
import cv2 # OpenCV
import matplotlib as mtplb
import numpy as np
import argparse as ap
import pathlib
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(".")

from tools.utils.process_text_file import ProcessTextFile
from tools.algo.perspective_projection import PerspectiveProjection
from tools.algo.dlt import DLT
from tools.utils.animation import Animation
from tools.geometry.rotations import Rotations
from tools.utils.image import Image
from tools.algo.harris import Harris

# Initial parameters
params = dict()
params["corner_patch_size"] = 9
params["harris_kappa"] = 0.08
params["num_keypoints"] = 200
params["nonmaximum_suppression_radius"] = 8
params["descriptor_radius"] = 9
params["match_lambda"] = 4

img = Image()
current_file_path = str(pathlib.Path(__file__).parent.absolute()) + "/data"
img.load_image(current_file_path + '/000000.png')

# Part 1 - Calculate Corner Response Functions
# Shi-Tomasi
harris = Harris(params)
score_shi_tomasi = harris.compute_shi_tomasi_score(img.gs)
score_shi_tomasi = Image.normalize(score_shi_tomasi)

# Harris
score_harris = harris.compute_haris_score(img.gs)
score_harris = Image.normalize(score_harris)

plt.figure(1)
plt.subplot(311)
plt.imshow(img.gs)

plt.subplot(312)
plt.imshow(score_shi_tomasi)

plt.subplot(313)
plt.imshow(score_harris)
plt.show()

# plt.pause(10)
plt.close()

# Part 2 - Select keypoints
keypoints = harris.select_keypoints(score_harris)

plt.figure(2)
plt.subplot(211)
plt.imshow(img.gs)

plt.subplot(212)
harris.plot_keypoints(keypoints)
plt.imshow(img.gs)
plt.show()

# plt.pause(10)
plt.close()

# Part 3 - Describe keypoints and show 16 strongest keypoint descriptors
descriptors = harris.get_keypoint_descriptors(keypoints, img.gs)

plt.figure(3)

# plot the 16 descriptors with the highest Harris score
for i in range(16):
    plt.subplot(4,4,i+1)
    descriptor = np.reshape(descriptors[i,:], (2*params["descriptor_radius"], 2*params["descriptor_radius"]))
    descriptor = Image.normalize(descriptor)
    plt.imshow(descriptor)

plt.show()
# plt.pause(1)
plt.close()

# Part 4 - Match descriptors beetween first two images
descriptors_frame_1 = descriptors

# get descriptors for frame 2
img2_path = current_file_path + '/000001.png'

keypoints_frame_2, descriptors_frame_2 = harris.get_keypoints_descriptors_from_image(img2_path)

img.load_image(img2_path)

matches = harris.match_descriptors(descriptors_frame_2, descriptors_frame_1)

plt.figure(4)
plt.imshow(img.gs)
harris.plot_keypoints(keypoints_frame_2)
harris.plot_matches(matches, keypoints_frame_2, keypoints)

plt.show()
# plt.pause(10)
plt.close()

# Part 5 - Match descriptors between all images
fig = plt.figure(5)
for i in range(199):
    img1_path = current_file_path + '/000' + "{0:0=3d}".format(i) + '.png'

    keypoints_frame_1, descriptors_frame_1 = harris.get_keypoints_descriptors_from_image(img1_path)

    img2_path = current_file_path + '/000' + "{0:0=3d}".format(i+1) + '.png'

    keypoints_frame_2, descriptors_frame_2 = harris.get_keypoints_descriptors_from_image(img2_path)

    matches = harris.match_descriptors(descriptors_frame_2, descriptors_frame_1)

    img.load_image(img2_path)
    fig.clf()
    plt.imshow(img.gs)
    harris.plot_keypoints(keypoints_frame_2)
    harris.plot_matches(matches, keypoints_frame_2, keypoints_frame_1)
    plt.show(block=False)
    plt.pause(0.001)

plt.close()