import os
import numpy as np
import cv2
import imutils
import pathlib
import sys
sys.path.append(".")

from tools.utils.image import Image
from tools.algo.sift import SIFT

params = dict()
params["rotation_inv"] = 1
rotation_img2_deg = 0

params["num_scales"] = 3 # scales per octave
params["kernel_size"] = (5,5)
params["num_octaves"] = 5 # number of octaves
params["sigma"] = 1.6
params["contrast_threshold"] = 0.04

image_file_1 = "images/img_1.jpg"
image_file_2 = "images/img_2.jpg"
rescale_factor = 0.2

current_file_path = str(pathlib.Path(__file__).parent.absolute()) + "/"

left_img = Image()
left_img.load_image(current_file_path + image_file_1)

right_img = Image()
right_img.load_image(current_file_path + image_file_2)

# To test rotational invariance of SIFT

if rotation_img2_deg != 0:

    right_img.gs = imutils.rotate(right_img.gs, rotation_img2_deg)

images = (left_img.gs, right_img.gs)

keypoint_locations = []
descriptors = []

sift = SIFT(params)
for img_idx in range(2):
    # 1) compute the image pyramid. Number of images in the pyramid equals "num_octaves".
    image_pyramid = sift.compute_image_pyramid(images[img_idx])

    # 2) Blur images for each octave. Each octave contains "num_scales + 3" blurred images.
    blurred_images = sift.compute_blurred_images(image_pyramid)

    # 3) Compute "num_scales + 2" difference of Gaussians for each octave
    DoGs = sift.compute_DoGs(blurred_images)

    # 4) Compute the keypoints with non-maximum supression and discard candidates with the contrast threshold.
    temp_keypoint_locations = sift.extract_keypoints_from_DoGs(DoGs)

    # 5) Given the blurred images and keypoints, compute the descriptors. Discard keypoints/descriptors that are too close to the boundary of the image. Some keypoints computed earlier are likely lost.
    