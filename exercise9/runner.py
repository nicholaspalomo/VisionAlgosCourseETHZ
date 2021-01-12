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
    pp_G_C = poses[:, [3, 7, 11]].transpose()

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
    T_V_C = np.reshape(hidden_state[0:num_frames*6], (6,-1), order='F')
    p_V_C = np.zeros((3, num_frames))
    for i in num_frames:
        single_T_V_C = BundleAdjustment.twist_2_homog_matrix(T_V_C[:,i])
        p_V_C[:,i] = single_T_V_C[:3, -1]

if __name__ == '__main__':

    main()