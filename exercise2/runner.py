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
        np.reshape(image_coordinates[0,:], (-1, 2), order='C'), 
        linestyle='o')
    plt.pause(0.1)
    plt.close()

    # Create an animation of the camera in 3D space
    rot_mat2 = np.array([0, -1, 0, 0, 0, -1, -1, 0, 0]).reshape((3,3))
    rot_mat3 = np.array([1, 0, 0, 0, 1, 0, 0, 0, -1]).reshape((3,3))
    rot_mat4 = np.matmul(rot_mat2, rot_mat3)

    R_C_W = dlt.M_tilde[:,:3].reshape((3,3,-1))
    t_C_W = dlt.M_tilde[:,-1].reshape((3,1,-1))
    quats = np.zeros((4,1,R_C_W.shape[2]))
    transl = np.zeros(t_C_W.shape)
    rot_mat = np.zeros(R_C_W.shape)
    for i in range(t_C_W.shape[2]):
        rot_mat[:,:,i] = np.matmul(rot_mat2, R_C_W[:,:,i]).transpose()
        transl[:,0,i] = np.matmul(-rot_mat[:,:,i], t_C_W[:,0,i])
        quats[:,:,i] = Rotations.rot_mat_2_quat(rot_mat[:,:,i]).reshape((4,1))

    # form a matrix of the quaternion rotations between the world coordinate frame and the camera
    p_W_corners = np.matmul(p_W_corners, rot_mat2.transpose())


    Animation.plot_trajectory_3D(15, transl.reshape((-1,3)), quats.reshape((-1,4)), p_W_corners)



if __name__ == "__main__":
    main()