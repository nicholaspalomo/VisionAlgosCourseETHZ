import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from parfor import parfor

from tools.utils.image import Image

class Stereo:
    def __init__(self, params, K, baseline):
        self.patch_radius = params['patch_radius']
        self.min_disp = params['min_disp']
        self.max_disp = params['max_disp']
        self.xlims = params['xlims']
        self.ylims = params['ylims']
        self.zlims = params['zlims']
        self.K = K
        self.baseline = baseline

    def get_disparity(self, left_img, right_img, debug_ssds=False, reject_outliers=True, refine_estimate=True):
        """
        left_img and right_img are both H x W and this method returns an H x W matrix containing the disparity, d, for each pixel of left_img.
        disp_img is set to 0 fir pixels where the SSD and/or d is not defined, and for d, estimates are rejected in Part 2. patch_radius specifies the SSD patch and each valid d should satisfy min_disp <= d <= max_disp
        """

        r = self.patch_radius
        patch_size = 2 * r + 1

        disp_img = np.zeros(left_img.shape)
        rows, cols = left_img.shape[0], left_img.shape[1]

        def dummy(*args):

            @parfor(args[0])
            def fun(row):
                if debug_ssds:
                    fig, ax = plt.subplots(1, 3)
    
                disp_img_col = np.zeros((cols-r-(self.max_disp+r),))
                for col in range(self.max_disp+r, cols-r):
                    left_patch = left_img[row-r:row+r+1, col-r:col+r+1]
                    right_strip = right_img[row-r:row+r+1, col-r-self.max_disp:col+r-self.min_disp+1]

                    # Transforming the patches into vectors so that they can be used with pdist2
                    lpvec = np.reshape(left_patch, (1,-1), order='F')
                    rsvecs = np.zeros((patch_size**2, self.max_disp - self.min_disp + 1))
                    for i in range(patch_size):
                        rsvecs[i*patch_size:(i+1)*patch_size, :] = right_strip[:, i:(self.max_disp - self.min_disp + i + 1)] # create a matrix of sub-patches of right_strip in order to find which disparity best describes the right_strip under consideration

                    ssds = cdist(lpvec, rsvecs.transpose(), metric='sqeuclidean').transpose()

                    if debug_ssds:
                        ax[0].cla()
                        ax[1].cla()
                        ax[2].cla()

                        ax[0].imshow((left_patch - left_patch.min()) / (left_patch.max() - left_patch.min() + 1e-5))
                        ax[0].axis('equal')

                        ax[1].imshow((right_strip - right_strip.min()) / (right_strip.max() - right_strip.min() + 1e-5))
                        ax[1].axis('equal')

                        ax[2].plot(ssds, marker='x')
                        plt.xlabel('d-dmax')
                        plt.ylabel('SSD(d)')

                        plt.pause(0.01)
                        
                    # With the current arrangement of the patches, the argmin of the ssds is not directly the disparity, but rather (max_disparity - disparity), referred to as the "neg_disp"

                    min_ssd = np.min(ssds)
                    neg_disp = np.argmin(ssds)

                    if reject_outliers:
                        if np.count_nonzero(ssds <= 1.5 * min_ssd) < 3 and neg_disp != 0 and neg_disp != ssds.shape[0]-1:
                            if not refine_estimate:
                                disp_img_col[col - (self.max_disp+r)] = self.max_disp - neg_disp
                            else:
                                x = np.array([neg_disp-1, neg_disp, neg_disp+1])
                                p = np.polyfit(x, ssds[x[:]], 2)
                                # Minimum of p(1)x^2 + p(2)x + p(3), converted from neg_disp to disparity as above

                                disp_img_col[col - (self.max_disp+r)] = self.max_disp + p[1]/(2 * p[0])
                    else:
                        disp_img_col[col - (self.max_disp+r)] = self.max_disp - neg_disp

                if debug_ssds:
                    plt.close()
            
                return disp_img_col # disp_img

            return fun

        disp_img_cols = dummy(range(r, rows-r))

        for i in range(r, rows-r):
            disp_img[i,self.max_disp+r:cols-r] = disp_img_cols[i-r]

        return disp_img

    def disparity_to_point_cloud(self, disp_img, left_img):
        """
        Points should be Nx3 and intensities Nx1 where N is the number of pixels which have a valid disparity. This method returns only points and intensities for pixels of left_img which have a valid disparity estimate.
        The i-th intensity corresponds to the i-th point.
        """

        xy = Image.meshgrid(disp_img.shape[1], disp_img.shape[0])

        px_left = np.concatenate(\
            (xy[:,:,1].reshape((-1,1), order='F'),\
            xy[:,:,0].reshape((-1,1), order='F'),\
            xy[:,:,2].reshape((-1,1), order='F')),\
            axis=1)

        # Corresponding pixels in right image = pixel coords in left image minus disparity
        px_right = px_left.copy()
        px_right[:,1] -= np.reshape(disp_img, (-1,), order='F')

        # Switch from (row, col, 1) to (u, v, 1)
        px_left[:,:2] = np.fliplr(px_left[:,:2])
        px_right[:,:2] = np.fliplr(px_right[:,:2])

        # Filter out pixels that do not have a valid disparity
        px_left = px_left[disp_img.reshape((-1,), order='F') > 0, :]
        px_right = px_right[disp_img.reshape((-1,), order='F') > 0, :]

        # Reproject pixels: Get bearing vectors of rays in camera frame
        bv_left = np.matmul(np.linalg.inv(self.K), px_left.transpose()).transpose()
        bv_right = np.matmul(np.linalg.inv(self.K), px_right.transpose()).transpose()

        # Intersect rays according to formula in problem statement
        points = np.zeros(px_left.shape)
        b = np.array([self.baseline, 0, 0]).reshape((3,1))
        for i in range(px_left.shape[0]):
            A = np.concatenate((bv_left[i,:], -bv_right[i,:]), axis=0).reshape((3,2), order='F')
            lamda = np.matmul(\
                np.linalg.inv(np.matmul(A.transpose(), A)), 
                np.matmul(A.transpose(), b))
            points[i,:] = bv_left[i,:] * lamda[0]

        intensities = left_img.reshape((-1,), order='F')[disp_img.reshape((-1,), order='F') > 0]

        return points, intensities # return the points given in X-Y-Z of the CAMERA frame