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

    image_coordinates = dlt.reproject_points(detected_corners, p_W_corners)

    # Show the corner projects projected in the image
    plt.figure()
    dlt.draw_line_in_I_from_points_in_I(
        current_file_path + "/images_undistorted/img_0001.jpg",
        image_coordinates[:,:,0], 
        linestyle='o')
    plt.pause(1)
    plt.close()

    # Create an animation of the camera in 3D space
    quats = np.zeros((detected_corners.shape[0],4))
    transl = np.zeros((detected_corners.shape[0],3))
    rot_mat = np.zeros((3,3,detected_corners.shape[0]))
    for i in range(detected_corners.shape[0]):
        rot_mat[:,:,i] = dlt.R_W_C[:,:,i].transpose()
        transl[i,:] = np.matmul(-dlt.R_W_C[:,:,i].transpose(), dlt.t_W_C[i,:])
        quats[i,:] = Rotations.rot_mat_2_quat(rot_mat[:,:,i])
        
    Animation.plot_trajectory_3D(30, transl, quats, p_W_corners)

if __name__ == "__main__":
    main()