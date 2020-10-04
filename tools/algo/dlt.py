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
        self.M_ = np.zeros((3,3))
        self.alpha_ = 1.

    def estimate_pose_dlt(self, p, P):
        """
        Inputs: p - 2D correspondence point
                P - 3D correspondence point
        """
        K = self.perspective_projection.camera_K_matrix_
        K_inv = np.linalg.inv(K)
        
        p_ = p.copy()
        p_ = np.reshape(p_, (-1, 2))
        p_ = np.concatenate((p_, np.ones((p_.shape[0], 1))), axis=1)

        normalized_coordinates = np.matmul(p_, K_inv.transpose())

        P_ = P.copy()
        P_ = np.concatenate((P_, np.ones((P_.shape[0], 1))), axis=1)

        Q = np.zeros((2 * 12, 12))
        j = 0
        for i in range(12):
            Q[j, 0:4] = P_[i,:]
            Q[j+1, 4:8] = P_[i,:]
            Q[j:j+1, 8:12] = np.expand_dims(P_[i,:], axis=0) *  -normalized_coordinates[i, 0]
            Q[j+1:j+2, 8:12] = np.expand_dims(P_[i,:], axis=0) * -normalized_coordinates[i, 1]
            j += 2

        U, _, V_transpose = np.linalg.svd(Q, full_matrices=True)

        self.M_ = np.reshape(V_transpose[-1,:], (4,3)).transpose() # double check the ordering here to make sure it is correct
        if self.M_[2, 3] < 0:
            self.M_ *= -1

        rot_mat = PerspectiveProjection.orthonormal_mat(self.M_[:,:3])

        self.alpha_ = np.linalg.norm(rot_mat) / np.linalg.norm(self.M_[:,:3])

        self.M_[:,:3] = rot_mat

        return self.M_, self.alpha_

    def reproject_points(self, p, P):
        M, alpha = self.estimate_pose_dlt(p, P)

        camera_pose = DLT.rot_mat_2_angle_axis(M[:3,:3])
        camera_pose = np.concatenate((camera_pose, 1/alpha * M[:,3,np.newaxis]), axis=0)

        pixel_coordinates = np.zeros((P.shape[0], 2))
        for i in range(P.shape[0]):
            coords, _ = self.perspective_projection.project_W_to_I(camera_pose, np.reshape(P[i,:], (3, 1)))
            pixel_coordinates[i, :] = np.reshape(coords, (1,2)).astype(int)

        return pixel_coordinates

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