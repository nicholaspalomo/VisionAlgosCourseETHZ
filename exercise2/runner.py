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

def main():
    current_file_path = str(pathlib.Path(__file__).parent.absolute()) + "/data"

    text_file_handler = ProcessTextFile()
    camera_K_matrix = text_file_handler.read_file(current_file_path + "/K.txt")
    p_W_corners = text_file_handler.read_file(current_file_path + "/p_W_corners.txt") * 0.01 # convert from cm to m
    detected_corners = text_file_handler.read_file(current_file_path + "/detected_corners.txt")

    dlt = DLT(camera_K_matrix, np.array([0., 0.]))

    image_coordinates = np.zeros((p_W_corners.shape[0], 2, detected_corners.shape[0]))
    R_W_C = np.zeros((3, 3, detected_corners.shape[0]))
    t_W_C = np.zeros((3, 1, detected_corners.shape[0]))
    for row in range(detected_corners.shape[0]):
        corners = np.reshape(detected_corners[row, :], (-1, 2))
        image_coordinates[:, :, row] = dlt.reproject_points(corners, p_W_corners)
        R_W_C[:, :, row] = dlt.R_W_C
        t_W_C[:, :, row] = dlt.t_W_C

    # Show the corner projects projected in the image
    plt.figure()
    dlt.draw_line_in_I_from_points_in_I(
        current_file_path + "/images_undistorted/img_0001.jpg",
        image_coordinates[:,:,0], 
        linestyle='o')
    plt.pause(1)
    plt.close()

    # Create an animation of the camera in 3D space
    rot_mat2 = np.array([-1, 0, 0, 0, 0, -1, 0, -1, 0]).reshape((3,3)) # additional rotation to align W coordinate frame given in figure 1 of the assignment with the standard x-y-z frame in which the x-y axes are in the ground plane and the z-axis is normal

    quats = np.zeros((detected_corners.shape[0],4))
    transl = np.zeros((detected_corners.shape[0],3))
    rot_mat = np.zeros((3,3,detected_corners.shape[0]))
    for i in range(detected_corners.shape[0]):
        rot_mat[:,:,i] = np.matmul(rot_mat2, R_W_C[:,:,i].transpose())
        transl[i,:] = np.matmul(-rot_mat[:,:,i], t_W_C[:,:,i]).squeeze(axis=1)
        quats[i,:] = Rotations.rot_mat_2_quat(rot_mat[:,:,i])

    p_W_corners = np.matmul(p_W_corners, rot_mat2.transpose())
        
    Animation.plot_trajectory_3D(30, transl, quats, p_W_corners)

if __name__ == "__main__":
    main()