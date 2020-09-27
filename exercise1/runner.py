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
    perspective_projection.load_image(current_file_path + "/images_undistorted/img_0001.jpg")

    # Exercise 1, 2.2
    # project the world origin to the camera/image frame
    delta = 0.04 # cm spacing between corners
    point_in_W = np.zeros((3,1))

    def grid(camera_pose):
        corners_x, corners_y = np.meshgrid(np.linspace(0, 8 * delta, 9), np.linspace(0, 5 * delta, 6))
        corners_in_C = np.zeros((corners_x.shape[0]*corners_x.shape[1], 2))
        k = 0
        for i in range(corners_x.shape[0]):
            for j in range(corners_x.shape[1]):
                point_in_W[0] = corners_x[i, j]
                point_in_W[1] = corners_y[i, j]
                point_in_C, _ = perspective_projection.project_W_to_C(camera_pose, point_in_W)
                corners_in_C[k, :] = np.transpose(point_in_C)
                k += 1

        return corners_in_C

    plt.figure()
    perspective_projection.line(grid, camera_poses[0,:], 'o')
    perspective_projection.display_image()
    plt.pause(3)
    plt.close()

    # Exercise 1, 2.3
    def get_cube_corners_in_C(camera_pose):
        edge_length = 2.
        corners_x = 4. + edge_length * np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0])
        corners_y = 1. + edge_length * np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        corners_z = 0. - edge_length * np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        corners_in_C = np.zeros((len(corners_x), 2))
        m = 0
        for i, j, k in zip(corners_x, corners_y, corners_z):
            point_in_W[0], point_in_W[1], point_in_W[2] = i, j, k
            point_in_C, _ = perspective_projection.project_W_to_C(camera_pose, delta * point_in_W)
            corners_in_C[m, :] = np.transpose(point_in_C)
            m += 1

        return corners_in_C

    plt.figure()
    perspective_projection.line(get_cube_corners_in_C, camera_poses[0,:], '-o')
    plt.pause(3)
    plt.close()

    # Exercise 1, 2.4
    plt.figure()
    perspective_projection.animation(current_file_path + "/images", get_cube_corners_in_C, camera_poses, '-o')
    plt.close()

    # Exercise 1, 3.1




if __name__ == "__main__":
    main()