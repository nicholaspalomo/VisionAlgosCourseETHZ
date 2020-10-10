from operator import matmul
from os import stat
from matplotlib import image
import numpy as np
import cv2 # OpenCV
import math
from matplotlib import pyplot as plt
import os

from .perspective_projection import PerspectiveProjection

class DLT:
    def __init__(self, camera_K_matrix, camera_D_matrix):
        self.perspective_projection = PerspectiveProjection(camera_K_matrix, camera_D_matrix)
        self.M_tilde = np.zeros((3,4))
        self.alpha = 1.

    def estimate_pose_dlt(self, p, P):
        """
        Inputs: p - 2D correspondence point
                P - 3D correspondence point
        """
        K_inv = np.linalg.inv(self.perspective_projection.camera_K_matrix_)

        P_ = np.concatenate((P, np.ones((P.shape[0], 1))), axis=1)
        num_points = P_.shape[0]
        
        p_ = np.reshape(p, (-1, 2), order='A')
        p_ = np.concatenate((p_, np.ones((p_.shape[0], 1))), axis=1)

        normalized_coordinates = np.matmul(p_, K_inv.transpose())

        Q = np.zeros((2 * num_points, num_points))
        j = 0
        for i in range(num_points):
            P_i = P_[np.newaxis,i,:]
            Q[j, :] = np.concatenate((P_i, np.zeros((1,4)), P_i *  -normalized_coordinates[i, 0]), axis=1)
            Q[j+1, :] = np.concatenate((np.zeros((1,4)), P_i, P_i *  -normalized_coordinates[i, 1]), axis=1)
            j += 2

        _, _, Vt = np.linalg.svd(Q, full_matrices=True)

        self.M_tilde = np.reshape(Vt[-1,:], (3,4), order='A') # double check the ordering here to make sure it is correct
        if np.linalg.det(self.M_tilde[:,:3]) < 0:
            self.M_tilde *= -1

        rot_mat = PerspectiveProjection.orthonormal_mat(self.M_tilde[:,:3])

        self.alpha = np.linalg.norm(rot_mat) / np.linalg.norm(self.M_tilde[:,:3])

        self.M_tilde[:,:3] = rot_mat
        self.M_tilde[:,-1] *= self.alpha

        return self.M_tilde, self.alpha

    def reproject_points(self, p, P):
        M, _ = self.estimate_pose_dlt(p, P)

        K = self.perspective_projection.camera_K_matrix_

        P_ = np.concatenate((P, np.ones((P.shape[0],1))), axis=1)

        pixel_coordinates = np.matmul(K, np.matmul(M, P_.transpose())).transpose()
        pixel_coordinates[:,0] = np.divide(pixel_coordinates[:,0], pixel_coordinates[:,2])
        pixel_coordinates[:,1] = np.divide(pixel_coordinates[:,1], pixel_coordinates[:,2])

        return pixel_coordinates[:,:2]

    def draw_line_in_I_from_points_in_I(self, image_path, points_in_I, linestyle='o'):

        if points_in_I.shape[1] > 3:
            points_in_I = np.transpose(points_in_I)

        self.perspective_projection.load_image(image_path)

        plt.plot(points_in_I[:,0], points_in_I[:,1], linestyle)

        self.perspective_projection.display_image()

        return

    @staticmethod
    def rot_mat_2_angle_axis(rot_mat):
        theta = math.acos((np.trace(rot_mat) - 1) / 2)

        return 1 / (2 * math.sin(theta)) * np.array([rot_mat[2,1] - rot_mat[1,2], rot_mat[0,2] - rot_mat[2,0], rot_mat[1,0] - rot_mat[0,1]]).reshape((3,1))