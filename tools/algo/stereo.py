import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from parfor import parfor

class Stereo:
    def __init__(self, params):
        self.patch_radius = params['patch_radius']
        self.min_disp = params['min_disp']
        self.max_disp = params['max_disp']
        self.xlims = params['xlims']
        self.ylims = params['ylims']
        self.zlims = params['zlims']

    def get_disparity(self, left_img, right_img, debug_ssds=False, reject_outliers=True, refine_estimate=True):
        """
        left_img and right_img are both H x W and this method returns an H x W matrix containing the disparity, d, for each pixel of left_img.
        disp_img is set to 0 fir pixels where the SSD and/or d is not defined, and for d, estimates are rejected in Part 2. patch_radius specifies the SSD patch and each valid d should satisfy min_disp <= d <= max_disp
        """

        r = self.patch_radius
        patch_size = 2 * r + 1

        disp_img = np.zeros(left_img.shape)
        rows, cols = left_img.shape[0], left_img.shape[1]

        for row in range(r, rows-r): # todo(nico) : replace this for with parfor. See example in exercise 8
            for col in range(self.max_disp+r, cols-r):
                left_patch = left_img[row-r:row+r, col-r:col+r]
                right_strip = right_img[row-r:row+r, col-r-self.max_disp:col+r-self.min_disp]

                # Transforming the patches into vectors so that they can be used with pdist2
                lpvec = np.reshape(left_patch, (-1,))
                rsvecs = np.zeros((r**2, self.max_disp - self.min_disp + 1))
                for i in patch_size:
                    rsvecs[i*patch_size:(i+1)*patch_size, :] = right_strip[:, i:(self.max_disp - self.min_disp + i)]

                ssds = cdist(lpvec, rsvecs, metric='euclidean').transpose()



        return disp_img