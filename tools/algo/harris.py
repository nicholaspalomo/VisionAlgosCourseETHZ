from operator import matmul
from os import stat
from matplotlib import image
import numpy as np
from numpy.core.fromnumeric import sort
from scipy.spatial.distance import cdist
import cv2 # OpenCV
import math
from matplotlib import pyplot as plt
import os
from tools.utils.image import Image

class Harris:
    def __init__(self, params):
        self.corner_patch_size = params["corner_patch_size"]
        self.harris_kappa = params["harris_kappa"]
        self.num_keypoints = params["num_keypoints"]
        self.nonmaximum_suppression_radius = params["nonmaximum_suppression_radius"]
        self.descriptor_radius = params["descriptor_radius"]
        self.match_lambda = params["match_lambda"]

        self.sobel = {'x' : np.array([-1., 0., 1., -2., 0., 2., -1., 0., 1.]).reshape((3,3))\
            , 'y' : np.array([-1., -2., -1., 0., 0., 0., 1., 2., 1.]).reshape((3,3))}

    def compute_haris_score(self, img):

        sum_Ix_squared, sum_Iy_squared, sum_IxIy_squared = self.convolve_sobel_xy(img)

        R_harris = sum_Ix_squared * sum_Iy_squared - sum_IxIy_squared * sum_IxIy_squared - self.harris_kappa * (sum_Ix_squared + sum_Iy_squared)**2

        R_harris[R_harris < 0] = 0

        return R_harris

    def compute_shi_tomasi_score(self, img):

        sum_Ix_squared, sum_Iy_squared, sum_IxIy_squared = self.convolve_sobel_xy(img)

        trace = sum_Ix_squared + sum_Iy_squared

        determinant = sum_Ix_squared * sum_Iy_squared - sum_IxIy_squared**2

        R_shi_tomasi = trace / 2 - (trace**2 / 4 - determinant)**0.5

        R_shi_tomasi[R_shi_tomasi < 0] = 0

        return R_shi_tomasi

    def convolve_sobel_xy(self, img):

        img_normalized = img.copy() / np.max(img)

        Ix, Iy = cv2.filter2D(img_normalized, -1, self.sobel["x"], borderType=cv2.BORDER_CONSTANT), cv2.filter2D(img_normalized, -1, self.sobel["y"], borderType=cv2.BORDER_CONSTANT)

        Ix_squared, Iy_squared, IxIy = Ix**2, Iy**2, Ix * Iy

        patch = np.ones((self.corner_patch_size, self.corner_patch_size))

        return cv2.filter2D(Ix_squared, -1, patch, borderType=cv2.BORDER_CONSTANT), cv2.filter2D(Iy_squared, -1, patch, borderType=cv2.BORDER_CONSTANT), cv2.filter2D(IxIy, -1, patch, borderType=cv2.BORDER_CONSTANT)

    def convolve(self, signal, kernel):

        (signal_height, signal_width) = signal.shape[:2]
        (kernel_height, kernel_width) = kernel.shape[:2]

        padding = (kernel_width) // 2
        signal = cv2.copyMakeBorder(signal, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

        signal_convolved = np.zeros((signal_height, signal_width), dtype="float32")
        for row in range(padding, padding + signal_height):
            for col in range(padding, padding + signal_width):
                patch = signal[row - padding:row + padding + 1, col - padding:col + padding + 1]

                patch_convolved = np.sum(patch * kernel)

                signal_convolved[row - padding, col - padding] = patch_convolved

        return signal_convolved

    def select_keypoints(self, score_mat):

        r = self.nonmaximum_suppression_radius

        keypoints = np.zeros((self.num_keypoints ,2)).astype(int)

        padded_score_mat = np.array(Harris.padarray(score_mat, 2*r))

        for i in range(self.num_keypoints):
            kp = np.argmax(padded_score_mat)
            kp = Harris.sub2ind(padded_score_mat.shape, kp)
            keypoints[i,:] = np.asarray(kp).astype(int) - r
            padded_score_mat[kp[0]-r:kp[0]+r, kp[1]-r:kp[1]+r] = 0

        return keypoints

    def plot_keypoints(self, keypoints, linestyle="x", marker_size=3):

        keypoints_tmp = Harris.col2row_matrix(keypoints)

        plt.plot(keypoints_tmp[:,1], keypoints_tmp[:,0], marker=linestyle, linestyle=" ", markersize=marker_size, color="r")

        return

    def get_keypoint_descriptors(self, keypoints, img):

        keypoints_tmp = Harris.col2row_matrix(keypoints)

        r = self.descriptor_radius

        descriptors = np.zeros((keypoints_tmp.shape[0],(2*r)**2))
        
        padded_img = np.array(Harris.padarray(img, 2*r))

        for i in range(keypoints_tmp.shape[0]):
            kp = keypoints_tmp[i,:] + r
            descriptors[i,:] = np.reshape(padded_img[kp[0]-r:kp[0]+r, kp[1]-r:kp[1]+r], (1,-1))

        return descriptors

    def get_keypoints_descriptors_from_image(self, img_path):

        img = Image()
        img.load_image(img_path)

        score_harris = self.compute_haris_score(img.gs)
        score_harris = Image.normalize(score_harris)
        
        keypoints = self.select_keypoints(score_harris)

        return keypoints, self.get_keypoint_descriptors(keypoints, img.gs)

    def match_descriptors(self, query_descriptors, database_descriptors):

        lamda = self.match_lambda

        dists = cdist(database_descriptors, query_descriptors, metric='euclidean').transpose()
        matches = np.argmin(dists, axis=1)
        dists = np.min(dists, axis=1)

        sorted_dists = np.sort(dists)
        sorted_dists = sorted_dists[sorted_dists > 0]

        min_non_zero_dist = sorted_dists[0]

        matches[dists >= lamda * min_non_zero_dist] = 0

        # remove double matches
        _, unique_match_idxs = np.unique(matches, return_index=True)

        unique_matches = np.zeros(matches.shape)
        unique_matches[unique_match_idxs] = matches[unique_match_idxs]

        return unique_matches

    def plot_matches(self, matches, query_keypoints, database_keypoints):

        query_indices = np.argwhere(matches > 0)
        match_indices = matches[matches > 0].astype(int)

        x_from = np.reshape(query_keypoints[query_indices, 0], (-1,1))
        x_to = np.reshape(database_keypoints[match_indices, 0], (-1,1))
        y_from = np.reshape(query_keypoints[query_indices, 1], (-1,1))
        y_to = np.reshape(database_keypoints[match_indices, 1], (-1,1))

        for i in range(y_from.shape[0]):
            plt.plot(np.array([y_from[i], y_to[i]]), np.array([x_from[i], x_to[i]]), color='g', linestyle='-', linewidth=2)

        return

    @staticmethod
    def col2row_matrix(mat):
        
        mat_tmp = mat.copy()
        if mat.shape[1] > mat.shape[0]:
            mat_tmp = np.transpose(mat_tmp)

        return mat_tmp

    @staticmethod
    def padarray(array, corner_patch_size):

        array_padded = array.copy()

        patch_radius = math.floor(corner_patch_size / 2)

        array_padded = cv2.copyMakeBorder(array, patch_radius, patch_radius, patch_radius, patch_radius, cv2.BORDER_CONSTANT)

        return array_padded

    @staticmethod
    def sub2ind(sz, ind):

        row = ind // sz[1]
        col = ind % sz[1]

        return (row, col)