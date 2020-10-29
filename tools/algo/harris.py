from operator import matmul
from os import stat
from matplotlib import image
import numpy as np
import cv2 # OpenCV
import math
from matplotlib import pyplot as plt
import os

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

        R_harris = Harris.padarray(R_harris, self.corner_patch_size)

        return R_harris

    def compute_shi_tomasi_score(self, img):

        sum_Ix_squared, sum_Iy_squared, sum_IxIy_squared = self.convolve_sobel_xy(img)

        trace = sum_Ix_squared + sum_Iy_squared

        determinant = sum_Ix_squared * sum_Iy_squared - sum_IxIy_squared**2

        R_shi_tomasi = trace / 2 - (trace**2 / 4 - determinant)**0.5

        R_shi_tomasi[R_shi_tomasi < 0] = 0

        R_shi_tomasi = Harris.padarray(R_shi_tomasi, self.corner_patch_size)

        return R_shi_tomasi

    @staticmethod
    def padarray(array, corner_patch_size):

        patch_radius = math.floor(corner_patch_size / 2)

        cv2.copyMakeBorder(array, patch_radius, patch_radius, patch_radius, patch_radius, cv2.BORDER_CONSTANT)

        return array

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