import cv2 # OpenCV
import matplotlib as mtplb
import numpy as np
import argparse as ap
import pathlib
from matplotlib import pyplot as plt
import os

from tools.utils.process_text_file import ProcessTextFile
from tools.algo.perspective_projection import PerspectiveProjection

def main():
    current_file_path = str(pathlib.Path(__file__).parent.absolute()) + "/data"

    text_file_handler = ProcessTextFile()
    camera_K_matrix = text_file_handler.read_file(current_file_path + "/K.txt")
    camera_D_matrix = text_file_handler.read_file(current_file_path + "/D.txt")
    camera_poses = text_file_handler.read_file(current_file_path + "/poses.txt")

    perspective_projection = PerspectiveProjection(camera_K_matrix, camera_D_matrix)

    # Exercise 1, 2.2
    def grid():
        corners_x, corners_y = np.meshgrid(np.linspace(0, 8, 9), np.linspace(0, 5, 6))

        corners_in_W = np.zeros((corners_x.shape[0]*corners_x.shape[1], 3))
        corners_in_W[:,0] = np.reshape(corners_x, (corners_in_W.shape[0], 1)).squeeze(axis=1)
        corners_in_W[:,1] = np.reshape(corners_y, (corners_in_W.shape[0], 1)).squeeze(axis=1)

        return corners_in_W * 0.04

    plt.figure()
    perspective_projection.draw_line_in_I(
        current_file_path + "/images_undistorted/img_0001.jpg", 
        grid(), 
        camera_poses[0,:],
        linestyle='o',
        image_distortion=False)
    plt.pause(3)
    plt.close()

    # Exercise 1, 2.3
    def get_cube_corners_in_W():
        edge_length = 2.
        corners_x = 4. + edge_length * np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0])
        corners_y = 1. + edge_length * np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        corners_z = 0. - edge_length * np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])

        corners_in_W = np.zeros((corners_x.shape[0], 3))
        corners_in_W[:,0] = corners_x
        corners_in_W[:,1] = corners_y
        corners_in_W[:,2] = corners_z

        return corners_in_W * 0.04

    plt.figure()
    perspective_projection.draw_line_in_I(
        current_file_path + "/images_undistorted/img_0001.jpg",
        get_cube_corners_in_W(),
        camera_poses[0,:],
        linestyle='-o', 
        image_distortion=False)
    plt.pause(3)
    plt.close()

    # Exercise 1, 2.4
    # plt.figure()
    # perspective_projection.animation_with_drawing_in_I(
    #     current_file_path + "/images",
    #     get_cube_corners_in_W(),
    #     camera_poses,
    #     linestyle='-o', 
    #     image_distortion=False)
    # plt.close()

    # Exercise 1, 3.1-2
    plt.figure()
    perspective_projection.draw_line_in_I(
        current_file_path + "/images/img_0001.jpg",
        get_cube_corners_in_W(),
        camera_poses[0,:],
        linestyle='-o',
        image_distortion=True)
    perspective_projection.display_image()
    plt.pause(3)
    plt.close()

    plt.figure()
    perspective_projection.animation_with_drawing_in_I(
        current_file_path + "/images",
        get_cube_corners_in_W(),
        camera_poses,
        linestyle='-o', 
        image_distortion=True)
    plt.close()

    # Exercise 1, 3.3


if __name__ == "__main__":
    main()