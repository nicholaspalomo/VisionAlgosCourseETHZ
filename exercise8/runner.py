# Runner script for exercise 8 (KLT)

import matplotlib as mtplb
import numpy as np
import argparse as ap
import pathlib
from matplotlib import pyplot as plt
from parfor import parfor
import cv2 # OpenCV
import os
import sys
sys.path.append(".")

from tools.utils.image import Image
from tools.utils.process_text_file import ProcessTextFile
from tools.algo.klt import KLT

def track_keypoints(*args, **kwargs):
    """
    Parallelize keypoint tracking
    """
    @parfor(args[0], args=[args[1], args[2], args[3], args[4], args[5]])
    def fun(j, img_ref, img_warped, keypoints, r_T, num_iters):
        klt = KLT()
        keypoints_j = keypoints[:,j]
        W, _ = klt.track_klt(img_ref, img_warped, keypoints_j, r_T, num_iters, do_plot=False)

        return W[:, -1]

    return fun

def track_keypoints_robustly(*args, **kwargs):
    @parfor(args[0], args=[args[1], args[2], args[3], args[4], args[5], args[6]])
    def fun(j, img_prev, img_warped, keypoints, r_T, num_iters, lamda):
        klt = KLT()
        keypoints_j = keypoints[:,j]
        W, keep = klt.track_klt_robustly(img_prev, img_warped, keypoints_j, r_T, num_iters, lamda)

        return (W[:, -1], keep)

    return fun

def main():
    img = Image()
    imgs_path = str(pathlib.Path(__file__).parent.absolute()) + "/data"
    img.load_image(imgs_path + '/000000.png')

    klt = KLT()
    x_T = np.array([[899],[290]])
    r_T = 15
    r_D = 20
    num_iters = 50
    lamda = 0.1

    ## Part 1: Warping images
    # fig, ax = plt.subplots(2, 2)
    # ax[0,0].imshow(img.gs)
    # ax[0,0].set_title('Reference image')

    # W = klt.get_sim_warp(50, -30, 0, 1)
    # img_warped = klt.warp_image(img.gs, W)
    # ax[0,1].imshow(img_warped)
    # ax[0,1].set_title('Translation')

    # W = klt.get_sim_warp(0, 0, 10, 1)
    # img_warped = klt.warp_image(img.gs, W)
    # ax[1,0].imshow(img_warped)
    # ax[1,0].set_title('Rotation around upper left corner')

    # W = klt.get_sim_warp(0, 0, 0, 0.5)
    # img_warped = klt.warp_image(img.gs, W)
    # ax[1,1].imshow(img_warped)
    # ax[1,1].set_title('Zoom on upper left corner')

    # plt.show()

    ## Part 2: Warped patches and recovering a simple warp with brute force
    # fig, ax = plt.subplots(1, 2)
    # ax[0].get_xaxis().set_visible(False)
    # ax[0].get_yaxis().set_visible(False)

    # ax[1].get_xaxis().set_visible(False)
    # ax[1].get_yaxis().set_visible(False)

    # W0 = klt.get_sim_warp(0, 0, 0, 1)
    # template = klt.get_warped_patch(img.gs, W0, x_T, r_T)

    # ax[0].imshow(Image.normalize(template))
    # ax[0].set_title('Template')

    W = klt.get_sim_warp(10, 6, 0, 1)
    img_warped = klt.warp_image(img.gs, W)
    # dx, ssds = klt.track_brute_force(img.gs, img_warped, x_T, r_T, r_D)

    # ax[1].imshow(Image.normalize(ssds))
    # ax[1].set_title('SSDs')

    # plt.show()

    # print("Displacement best explained by (dx, dy) = ({}, {})".format(str(dx[0]), str(dx[1])))

    ## Part 3: Recovering the warp with KLT
    # W, p_hist = klt.track_klt(img.gs, img_warped, x_T, r_T, num_iters, do_plot=True)

    # print("Point moved by {}, should move by (-10, -6)".format(W[:,-1]))

    ## Part 4: Applying KLT to KITTI
    file = ProcessTextFile()
    img.gs = cv2.resize(img.gs, dsize=(int(0.25 * img.gs.shape[1]), int(0.25 * img.gs.shape[0])), interpolation=cv2.INTER_CUBIC)
    keypoints_rc = file.read_file(imgs_path + '/keypoints.txt') / 4
    keypoints = np.flipud(np.transpose(keypoints_rc[0:50, :]))

    # fig = plt.figure(constrained_layout=True)
    # plt.imshow(img.gs)
    # ax = plt.gca()
    # ax.scatter(keypoints[0,:], keypoints[1,:], c='r', marker='x')
    # plt.pause(0.1)

    # img_prev = img.gs.copy()
    # for i in range(1, 21):
    #     img.load_image(imgs_path + ('/%06d.png' % i))
    #     img.gs = cv2.resize(img.gs, dsize=(int(0.25 * img.gs.shape[1]), int(0.25 * img.gs.shape[0])), interpolation=cv2.INTER_CUBIC)
    #     plt.imshow(img.gs)

    #     warps = track_keypoints(range(keypoints.shape[1]), img_prev, img.gs, keypoints, r_T, num_iters)

    #     dkp = np.zeros(keypoints.shape)
    #     for j in range(keypoints.shape[1]):
    #         dkp[:, j] = warps[j]

    #     kp_old = keypoints.copy()
    #     keypoints = keypoints + dkp
    #     img_prev = img.gs.copy()
    #     ax.scatter(kp_old[0,:], kp_old[1,:], c='r', marker='x')

    #     klt.plot_matches(np.arange(0, keypoints.shape[1]), np.flipud(keypoints).transpose(), np.flipud(kp_old).transpose(), ax)

    #     plt.pause(0.1)

    # plt.close()

    ## Part 5: Outlier rejection with bidirectional error
    img.load_image(imgs_path + '/000000.png')
    img.gs = cv2.resize(img.gs, dsize=(int(0.25 * img.gs.shape[1]), int(0.25 * img.gs.shape[0])), interpolation=cv2.INTER_CUBIC)

    fig = plt.figure(constrained_layout=True)
    plt.imshow(img.gs)
    ax = plt.gca()
    ax.scatter(keypoints[0,:], keypoints[1,:], c='r', marker='x')    
    plt.pause(0.1)

    img_prev = img.gs.copy()
    for i in range(1, 21):
        img.load_image(imgs_path + ('/%06d.png' % i))
        img.gs = cv2.resize(img.gs, dsize=(int(0.25 * img.gs.shape[1]), int(0.25 * img.gs.shape[0])), interpolation=cv2.INTER_CUBIC)

        out = track_keypoints_robustly(range(keypoints.shape[1]), img_prev, img.gs, keypoints, r_T, num_iters, lamda)

        dkp = np.zeros((keypoints.shape))
        keep = np.ones((keypoints.shape[1]), dtype=bool)
        for j in range(keypoints.shape[1]):
            dkp[:,j] = out[j][0]
            keep[j] = out[j][1]

        kp_old = keypoints[:, keep].copy()
        keypoints = keypoints + dkp
        keypoints = keypoints[:, keep]

        plt.imshow(img.gs)
        ax.scatter(kp_old[0,:], kp_old[1,:], c='r', marker='x')

        klt.plot_matches(np.arange(0, keypoints.shape[1]), np.flipud(keypoints).transpose(), np.flipud(kp_old).transpose(), ax)

        img_prev = img.gs.copy()

        plt.pause(0.1)

    plt.close()

if __name__ == '__main__':

    main()