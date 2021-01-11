# Runner script for exercise 7 (RANSAC, Lecture 9)

import random
import numpy as np
import argparse as ap
import pathlib
from matplotlib import pyplot as plt
from parfor import parfor
import cv2 # OpenCV
import os
import sys
sys.path.append(".")

random.seed(2) # set the random seed for RANSAC

from tools.utils.image import Image
from tools.utils.process_text_file import ProcessTextFile
from tools.algo.ransac import RANSAC

def rms(x):

    return np.mean(x**2)**0.5

def main():

    ## Create data for parts 1 and 2
    num_inliers = 20
    num_outliers = 10
    noise_ratio = 0.1
    poly = np.random.rand(3, 1) # random second-order polynomial
    extremum = -poly[1] / (2 * poly[0])
    xstart = extremum - 0.5
    lowest = np.polyval(poly, extremum)
    highest = np.polyval(poly, xstart)
    xspan = 1
    yspan = highest - lowest
    max_noise = noise_ratio * yspan
    x = np.random.rand(1, num_inliers) + xstart
    y = np.polyval(poly, x)
    y += (np.random.rand(y.shape[0], y.shape[1]) - 0.5) * 2 * max_noise
    data = np.concatenate(( \
        np.concatenate((x, \
        np.random.rand(1, num_outliers) + xstart), axis=1), \
        np.concatenate((y, \
        np.random.rand(1, num_outliers) * yspan + lowest), axis=1)), axis=0)

    ## Create data for parts 3 and 4
    file = ProcessTextFile()
    path = str(pathlib.Path(__file__).parent.absolute()) + "/data"
    K = file.read_file(path + "/K.txt")
    keypoints = file.read_file(path + "/keypoints.txt")
    p_W_landmarks = file.read_file(path + "/p_W_landmarks.txt")

    ## Data for part 4
    img = Image()
    img.load_image(path + '/000000.png')
    database_image = img.gs

    ## Dependencies
    from tools.algo.dlt import DLT # DLT implementation
    from tools.algo.harris import Harris # for using the keypoint/descriptor matcher

    ## Part 1 - RANSAC with parabola model
    ransac = RANSAC()
    best_guess_history, max_num_inliers_history = ransac.parabola_ransac(data, max_noise)

    # Compare with full data fit
    full_fit = np.polyfit(data[0, :], data[1, :], 2)

    _, ax = plt.subplots(1, 2)
    ax[0].scatter(data[0, :], data[1, :], color='b')

    x = np.arange(start=xstart, stop=(xstart + 1.), step=0.01)
    for i in range(best_guess_history.shape[1]):
        guess_plot, = ax[0].plot(x, np.polyval(best_guess_history[:, i], x), color='b')

    truth_plot, = ax[0].plot(x, np.polyval(poly, x), color='g', linewidth=2)

    best_plot, = ax[0].plot(x, np.polyval(best_guess_history[:, -1], x), color='r', linewidth=2)

    fit_plot, = ax[0].plot(x, np.polyval(full_fit, x), color='r', marker=".")

    ax[0].set_xlim(xstart, xstart+1)
    ax[0].set_ylim(lowest-max_noise, highest+max_noise)

    ax[0].legend((truth_plot, best_plot, fit_plot, guess_plot), ('ground truth', 'RANSAC result', 'full data fit', 'RANSAC guesses'))

    ax[0].set_title('RANSAC vs full fit')

    ax[1].plot(np.arange(0, max_num_inliers_history.shape[1], 1), max_num_inliers_history.squeeze(axis=0))

    ax[1].set_title('Max num inliers vs iterations')

    plt.show()
    plt.pause(0.1)
    plt.close()

    x = xstart + np.arange(0, 1, 0.01)
    print("RMS of full fit = {}".format(rms(np.polyval(poly, x) - np.polyval(full_fit, x))))

    print("RMS of RANSAC = {}".format(rms(np.polyval(poly, x) - np.polyval(best_guess_history[:, -1], x))))

    # Parts 2 and 3 - Localization with RANSAC + DLT/P3P
    query_image_path = path + '/000001.png'

    # Parameters from exercise 3 (DLT/PnP)
    params = dict()
    params["corner_patch_size"] = 9
    params["harris_kappa"] = 0.08
    params["num_keypoints"] = 1000
    params["nonmaximum_suppression_radius"] = 8
    params["descriptor_radius"] = 9
    params["match_lambda"] = 5

    # Run ransac with DLT
    R_C_W, t_C_W, inlier_mask, max_num_inliers_history, num_iteration_history = ransac.detect_localize_landmarks_ransac(p_W_landmarks, K, params, img.gs, keypoints, query_image_path, use_p3p=False)

    T_C_W = np.concatenate((R_C_W, t_C_W[:,np.newaxis]), axis=1)
    T_C_W = np.concatenate((T_C_W, np.array([0, 0, 0, 1])[np.newaxis,:]), axis=0)
    print("Found transformation T_C_W =\n {}".format(T_C_W))

    print("Estimated inlier ratio is {}".format(np.count_nonzero(inlier_mask)/inlier_mask.shape[0]))

    _, ax = plt.subplots(1, 1)
    ax.semilogy(np.arange(1, num_iteration_history.shape[0]+1), num_iteration_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Estimated Max Number of Iterations")

    plt.show()
    plt.pause(0.1)
    plt.close()

    _, ax = plt.subplots(3, 1)

    img = Image()
    img.load_image(query_image_path)
    ax[0].imshow(img.gs)
    ax[0].plot(ransac.query_keypoints[:,1], ransac.query_keypoints[:,0], c='r', marker='x', linewidth=2, linestyle=" ")
    ransac.plot_matches(ax=ax[0])
    ax[0].set_title("All keypoints and matches")

    ax[1].imshow(img.gs)
    ax[1].plot(ransac.matched_query_keypoints[(1-inlier_mask)>0, 1], ransac.matched_query_keypoints[(1-inlier_mask)>0, 0], c='r', marker='x', linewidth=2, linestyle=" ")
    ax[1].plot(ransac.matched_query_keypoints[inlier_mask>0, 1], ransac.matched_query_keypoints[inlier_mask>0, 0], c='g', marker='x', linestyle=" ")
    ransac.plot_matches(ax=ax[1], mask=True)
    ax[1].set_title("Inlier and outlier matches")

    ax[2].plot(np.arange(1, max_num_inliers_history.shape[0]+1), max_num_inliers_history)
    ax[2].set_title("Maximum inlier count over RANSAC iterations")

    plt.show()
    plt.pause(0.1)
    plt.close()

    ## Part 4 - Repeat the previous part, but for all frames

if __name__ == '__main__':

    main()