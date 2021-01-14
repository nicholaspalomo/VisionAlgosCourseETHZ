# Runner script for exercise 9 (Bundle Adjustment, Lecture 13)

from cv2 import data
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import argparse as ap
import pathlib
import cv2
import os
import sys

from numpy.core.shape_base import block
sys.path.append(".")

from tools.utils.image import Image
from tools.utils.process_text_file import ProcessTextFile
from tools.algo.perspective_projection import PerspectiveProjection
from tools.algo.bundle_adjustment import BundleAdjustment

def main():
    data_path = str(pathlib.Path(__file__).parent.absolute()) + "/data"
    file = ProcessTextFile()
    hidden_state = file.read_file(data_path + '/hidden_state.txt')
    observations = file.read_file(data_path + '/observations.txt')
    num_frames = 150
    K = file.read_file(data_path + '/K.txt')
    poses = file.read_file(data_path + '/poses.txt')
    # 'pp' stands for 'p prime'
    pp_G_C = poses[:, [3, 7, 11]]

    ba = BundleAdjustment()
    hidden_state, observations, pp_G_C = ba.crop_problem(\
        hidden_state, \
        observations, \
        pp_G_C, \
        num_frames)

    cropped_hidden_state, cropped_observations, _ = ba.crop_problem(\
        hidden_state, \
        observations, \
        pp_G_C, \
        4)

    ## Compare trajectory to ground truth
    # Remember, V is the "world frame of the visual odometry"...
    T_V_C = np.reshape(hidden_state[:num_frames*6], (6,-1), order='F')
    p_V_C = np.zeros((num_frames, 3))
    for i in range(num_frames):
        single_T_V_C = BundleAdjustment.twist_2_homog_matrix(T_V_C[:,i])
        p_V_C[i,:] = single_T_V_C[:3, -1]

    _, ax = plt.subplots(1, 1)
    # ... and G the "world frame of the ground truth"
    ax.scatter(pp_G_C[:, 2], -pp_G_C[:, 0], s=2)
    ax.scatter(p_V_C[:, 2], -p_V_C[:, 0], s=2)
    ax.axis('equal')
    ax.axis([-5, 95, -30, 10])
    plt.legend(['Ground truth', 'Estimate'])

    plt.show(block='False')
    plt.pause(1)
    plt.close()

    ## Align estimate to ground truth
    p_G_C = ba.align_estimate_to_ground_truth(pp_G_C, p_V_C)

    _, ax = plt.subplots(1, 1)
    ax.scatter(pp_G_C[:, 2], -pp_G_C[:, 0], s=2)
    ax.scatter(p_V_C[:, 2], -p_V_C[:, 0], s=2)
    ax.scatter(p_G_C[:, 2], -p_G_C[:, 0], s=2)
    ax.axis('equal')
    ax.axis([-5, 95, -30, 10])
    plt.legend(['Ground truth', 'Original estimate', 'Aligned estimate'])

    plt.show(block='False')
    plt.pause(1)
    plt.close()

    # Plot the state before bundle adjustment
    _, ax = plt.subplots(1, 1)
    ba.plot_map(cropped_hidden_state, cropped_observations, [0, 20, -5, 5], ax=ax)
    ax.set_title('Cropped problem before bundle adjustment')

    plt.show(block='False')
    plt.pause(1)
    plt.close()

if __name__ == '__main__':

    main()