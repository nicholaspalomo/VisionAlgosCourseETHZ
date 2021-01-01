from operator import matmul
from os import stat
from matplotlib import image
import numpy as np
import cv2 # OpenCV
import math
from matplotlib import pyplot as plt
import os
from scipy.signal import convolve2d

from .perspective_projection import PerspectiveProjection

class KLT:
    def __init__(self):

        return

    def get_sim_warp(self, dx, dy, alpha_deg, lamda):
        """
        Get the warp matrix
        """
        W = np.zeros((2,3))
        alpha_rad = alpha_deg * np.pi / 180
        c = math.cos(alpha_rad)
        s = math.sin(alpha_rad)
        W[:,:2] = np.array([[c, -s], [s, c]])
        W[:,-1] = np.array([[dx],[dy]]).squeeze(axis=1)
        W *= lamda

        return W.copy()

    def warp_image(self, img, W):

        h, w = img.shape

        img_warped = np.zeros(img.shape)

        for x in range(w):
            for y in range(h):
                warped = np.matmul(W, np.array([[x],[y],[1]])).squeeze(axis=1) # x,y
                if warped[0] < w-1 and warped[1] < h-1 and warped[0] > 0 and warped[1] > 0:
                    img_warped[y, x] = img[int(warped[1]), int(warped[0])].copy()

        return img_warped

    def get_warped_patch(self, img, W, x_T, r_T, interpolate=True):
        """
        Compute the warped image patch.
        Inputs:
        img - image containing a patch to warp
        x_T - coordinate of the template
        r_T - radius of warp
        Outputs:
        patch_warped - warped image patch (2*r_T+1)x(2*r_T+1)
        """

        patch_h = 2*r_T+1
        patch_w = patch_h

        patch_warped = np.zeros((patch_h, patch_w))

        h, w = img.shape
        Wt = np.transpose(W)

        for x in range(-r_T, r_T+1):
            for y in range(-r_T, r_T+1):
                pre_warp = np.expand_dims(np.array([x, y, 1]), axis=1).transpose()
                warped = (np.transpose(x_T) + np.matmul(pre_warp, Wt)).squeeze(axis=0)
                if warped[0] < w-1 and warped[1] < h-1 and warped[0] > 0 and warped[1] > 0:
                    #Don't forget that patch coefficients are 1:(2*r_T+1), rather than -r_T:r_T.
                    if interpolate:
                        floors = np.floor(warped)
                        weights = warped - floors
                        a = weights[0]
                        b = weights[1]

                        # bilinearly interpolate the warped patch
                        intensity = (1-b) * ( \
                            (1-a) * img[int(floors[1]), int(floors[0])] + \
                            a * img[int(floors[1]), int(floors[0]+1)]) + \
                            b * ( \
                            (1-a) * img[int(floors[1]+1), int(floors[0])] + \
                            a * img[int(floors[1]+1), int(floors[0]+1)])

                        patch_warped[y + r_T, x + r_T] = intensity

                    else:
                        patch_warped[y + r_T, x + r_T] = img[int(warped[1]), int(warped[0])]
                        
        return patch_warped

    def track_brute_force(self, img, img_warped, x_T, r_T, r_D):
        """
        Method to recover translation-only warp of a template

        Inputs:
        img - unwarped image
        img_warped - warped image
        x_T - point to track
        r_T - radius of region within which to search for warped patch
        num_iters - number of iterations
        r_D - patch radius

        Outputs:
        dx - translation that best explains where x_T is in image I
        ssds - SSDs for all values of dx within the patch defined by center c_T and radius r_D
        """

        ssds = np.zeros((2*r_D+1, 2*r_D+1))

        W = self.get_sim_warp(0, 0, 0, 1)
        template = self.get_warped_patch(img, W, x_T, r_T)

        for dx in range(-r_D, r_D+1):
            for dy in range(-r_D, r_D+1):
                W = self.get_sim_warp(dx, dy, 0, 1)
                candidate = self.get_warped_patch(img_warped, W, x_T, r_T)
                ssd = np.sum(np.square(template - candidate))
                ssds[int(dx + r_D), int(dy + r_D)] = ssd

        idx = np.argmin(ssds)
        dx1, dx2 = np.unravel_index(idx, ssds.shape, order='F')

        return np.array([dx1, dx2]) - r_D, ssds

    def track_klt(self, img, img_warped, x_T, r_T, num_iters, do_plot=True):
        """
        Method to track a patch using the Lucas-Kanade-Tomasi tracker

        Inputs:
        img - reference image
        img_warped - warped image
        x_T - point to track
        r_T - radius of patch to track
        num_iters - number of iterations

        Outputs:
        W - estimate of the warp parameters
        p_history - history of p-estimates, including the initial estimate (identity)
        """

        p_hist = np.zeros((6, num_iters+1))
        W = self.get_sim_warp(0, 0, 0, 1) # identity warp
        p_hist[:, 0] = np.reshape(W ,(6,), order='F')

        # T suffix indicates image evaluated for patch T
        patch_T = self.get_warped_patch(img, W, x_T, r_T) # patch of reference image
        img_R = np.reshape(patch_T, (patch_T.shape[0]*patch_T.shape[1], 1), order='F') # the template vector, img_R, never changes

        # x and y coordinates of the patch also never change
        xs = np.arange(-r_T, r_T+1)
        ys = xs.copy()
        n = xs.shape[0]
        xy1 = np.concatenate(( \
            np.kron(xs, np.ones((1, n))).transpose(), \
            np.kron(np.ones((1, n)), ys).transpose(), \
            np.ones((n*n, 1))
            ), axis=1)
        dwdp = np.kron(xy1, np.eye(2))

        if do_plot:
            fig, ax = plt.subplots(3, 1)

        kernel = np.array([1, 0, -1])
        for it in range(num_iters):
            big_IWT = self.get_warped_patch(img_warped, W, x_T, r_T + 1)
            IWT = big_IWT[1:-1, 1:-1]
            i = np.reshape(IWT, (-1,1), order='F')

            # computing di/dp
            IWTx = KLT.conv2(1, kernel, big_IWT[1:-1, :], mode='valid') # todo(nico): I don't think you can use cv2.filter2D here because you need to convolve with one kernel in one direction and with a second kernel in the other direction. To mimic the functionality of matlab's conv2, first convolve in the row direction (0) and then in the column direction (1)
            IWTy = KLT.conv2(kernel, 1, big_IWT[:, 1:-1], mode='valid')

            didw = np.concatenate((np.reshape(IWTx, (-1,1), order='F'), np.reshape(IWTy, (-1,1), order='F')), axis=1)
            didp = np.zeros((n*n, 6))
            for pixel_i in range(n*n):
                didp[pixel_i, :] = np.matmul(didw[pixel_i, :], dwdp[pixel_i*2:pixel_i*2+2, :])

            # computing hession
            H = np.matmul(np.transpose(didp), didp)
            
            # take gradient step
            delta_p = np.matmul(np.matmul(np.linalg.inv(H), np.transpose(didp)), img_R - i)
            W += np.reshape(delta_p, (2, 3), order='F')

            if do_plot:
                # plt.clf()
                tmp_mat = np.concatenate((IWT, patch_T, (patch_T - IWT)), axis=1)
                ax[0].imshow(tmp_mat)
                ax[0].set_title('I(W(T)), I_R(T) and the difference')

                tmp_mat = np.concatenate((IWTx, IWTy), axis=1)
                ax[1].imshow(tmp_mat)
                ax[1].set_title('warped gradients')

                descentcat = np.zeros((n, 6*n))
                for j in range(6):
                    descentcat[:, j*n:(j+1)*n] = np.reshape(didp[:,j], (n, n))
                ax[2].imshow(descentcat)
                ax[2].set_title('steepest descent images')
                plt.pause(0.1)
                # plt.close()

            p_hist[:, it + 1] = np.reshape(W, (-1,))

            if np.linalg.norm(delta_p) < 1e-3:
                p_hist = p_hist[:, :it+1]
                break

        return W, p_hist

    @staticmethod
    def conv2(v1, v2, m, mode='same'):
        """
        Two-dimensional convolution of matrix m by vectors v1 and v2

        First convolves each column of 'm' with the vector 'v1'
        and then it convolves each row of the result with the vector 'v2'.

        """
        tmp = np.apply_along_axis(np.convolve, 0, m, v1, mode)
        return np.apply_along_axis(np.convolve, 1, tmp, v2, mode)