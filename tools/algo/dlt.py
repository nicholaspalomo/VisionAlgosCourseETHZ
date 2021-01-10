from operator import matmul
from os import stat
from matplotlib import image
import numpy as np
from numpy import matlib
import cv2 # OpenCV
import math
from matplotlib import pyplot as plt
import os

from .perspective_projection import PerspectiveProjection

class DLT:
    def __init__(self, camera_K_matrix, camera_D_matrix):
        self.perspective_projection = PerspectiveProjection(camera_K_matrix, camera_D_matrix)
        self.M_tilde = None
        self.alpha = None
        self.R_W_C = None
        self.t_W_C = None

    def estimate_pose_dlt(self, p, P):
        """
        Inputs: p - 2D correspondence point
                P - 3D correspondence point
        """
        K_inv = np.linalg.inv(self.perspective_projection.camera_K_matrix_)

        P_ = np.concatenate((P, np.ones((P.shape[0], 1))), axis=1)
        num_points = P_.shape[0]
        
        p_ = np.concatenate((p, np.ones((p.shape[0], 1))), axis=1)

        normalized_coordinates = np.matmul(K_inv, p_.transpose()).transpose()

        Q = np.zeros((2 * num_points, 12))
        for k in range(P_.shape[0]):
            P_i = np.transpose(P_[k,:,np.newaxis])
            Q[2*k, :] = np.concatenate((P_i, np.zeros((1,4)), P_i *  -normalized_coordinates[k, 0]), axis=1)
            Q[2*k+1, :] = np.concatenate((np.zeros((1,4)), P_i, P_i *  -normalized_coordinates[k, 1]), axis=1)

        _, _, Vt = np.linalg.svd(Q, full_matrices=True)

        self.M_tilde = np.reshape(np.transpose(Vt)[:,-1], (4,3), order='F').transpose()
        if np.linalg.det(self.M_tilde[:,:3]) < 0:
            self.M_tilde *= -1

        self.R_W_C = PerspectiveProjection.orthonormal_mat(self.M_tilde[:,:3].copy())

        self.alpha = np.linalg.norm(self.R_W_C) / np.linalg.norm(self.M_tilde[:,:3])

        self.t_W_C = self.alpha * self.M_tilde[:,-1, np.newaxis]

        self.M_tilde[:,:3] = self.R_W_C.copy()
        self.M_tilde[:,-1] = self.alpha * self.M_tilde[:,-1].copy()

        return self.M_tilde.copy(), self.alpha.copy()

    def reproject_points(self, p, P):
        M, _ = self.estimate_pose_dlt(p, P)

        K = self.perspective_projection.camera_K_matrix_

        P_ = np.concatenate((P, np.ones((P.shape[0],1))), axis=1)

        pixel_coordinates_unnormalized = np.matmul(K, np.matmul(M, P_.transpose())).transpose()

        pixel_coordinates = np.zeros((p.shape[0], 2))
        pixel_coordinates[:,0] = np.divide(pixel_coordinates_unnormalized[:,0], pixel_coordinates_unnormalized[:,2])

        pixel_coordinates[:,1] = np.divide(pixel_coordinates_unnormalized[:,1], pixel_coordinates_unnormalized[:,2])

        return pixel_coordinates

    def project_points(self, points_3d):

        K = self.perspective_projection.camera_K_matrix_

        # transform coordinates from camera frame (points_3d) to image plane
        projected_points = np.matmul(K, points_3d.transpose()).transpose()
        projected_points = np.divide(projected_points, np.matlib.repmat(projected_points[:,2], 3, 1).transpose())

        # apply distortion
        projected_points = self.perspective_projection.distort_points(projected_points[:,:2])

        return projected_points

    def draw_line_in_I_from_points_in_I(self, image_path, points_in_I, linestyle='o'):

        if points_in_I.shape[1] > 3:
            points_in_I = np.transpose(points_in_I)

        self.perspective_projection.load_image(image_path)

        plt.plot(points_in_I[:,0], points_in_I[:,1], linestyle)

        self.perspective_projection.display_image()

        return