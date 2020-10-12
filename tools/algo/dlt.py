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
        
        p_ = p[:,np.newaxis,:].transpose()
        p_ = np.reshape(p_, (int(p_.shape[0]/2), 2, -1), order='C')
        p_ = np.concatenate((p_, np.ones((p_.shape[0], 1, p_.shape[2]))), axis=1)

        normalized_coordinates = np.zeros(p_.shape)
        for i in range(p_.shape[2]):
            for j in range(p_.shape[0]):
                normalized_coordinates[j,:,i] = np.matmul(K_inv, p_[j,:,i].transpose()).transpose()

        Q = np.zeros((2 * num_points, num_points))

        M = np.zeros((3, 4, p.shape[0]))
        alphas = np.zeros((p.shape[0], 1))

        for k in range(p.shape[0]):
            j = 0
            for i in range(num_points):
                P_i = P_[np.newaxis,i,:]
                Q[j, :] = np.concatenate((P_i, np.zeros((1,4)), P_i *  -normalized_coordinates[i, 0, k]), axis=1)
                Q[j+1, :] = np.concatenate((np.zeros((1,4)), P_i, P_i *  -normalized_coordinates[i, 1, k]), axis=1)
                j += 2

            _, _, Vt = np.linalg.svd(Q, full_matrices=True)

            M_tilde = np.reshape(Vt[-1,:], (3,4), order='A')
            if np.linalg.det(M_tilde[:,:3]) < 0:
                M_tilde *= -1

            rot_mat = PerspectiveProjection.orthonormal_mat(M_tilde[:,:3])

            alpha = np.linalg.norm(rot_mat) / np.linalg.norm(M_tilde[:,:3])

            M[:,:3,k] = rot_mat
            M[:,-1,k] = M_tilde[:,-1] * alpha

            alphas[k] = alpha

        self.M_tilde = M
        self.alpha = alphas
        self.R_W_C = M[:,:3,:]
        self.t_W_C = M[:,-1,:].transpose()

        return M, alphas

    def reproject_points(self, p, P):
        M, _ = self.estimate_pose_dlt(p, P)

        K = self.perspective_projection.camera_K_matrix_

        P_ = np.concatenate((P, np.ones((P.shape[0],1))), axis=1)

        # pixel_coordinates_unnormalized = np.matmul(np.kron(np.eye(p.shape[0]), K), np.matmul(M, P_.transpose())).transpose()
        # pixel_coordinates_unnormalized = np.reshape(pixel_coordinates_unnormalized[np.newaxis,:,:], (-1, p.shape[0], 3), order='C').transpose()
        # pixel_coordinates = []
        # for i in range(pixel_coordinates_unnormalized.shape[2]):
        #     pixel_coordinates.append(np.divide(pixel_coordinates_unnormalized[0,:,i], pixel_coordinates_unnormalized[2,:,i]))
        #     pixel_coordinates.append(np.divide(pixel_coordinates_unnormalized[1,:,i], pixel_coordinates_unnormalized[2,:,i]))            

        # return np.array(pixel_coordinates).transpose()

        pixel_coordinates_unnormalized = np.zeros((P.shape[0],3,p.shape[0]))
        pixel_coordinates = np.zeros((P.shape[0],2,p.shape[0]))
        for i in range(p.shape[0]):
            pixel_coordinates_unnormalized[:,:,i] = np.matmul(K, np.matmul(M[:,:,i], P_.transpose())).transpose()

            pixel_coordinates[:,0,i] = np.divide(pixel_coordinates_unnormalized[:,0,i], pixel_coordinates_unnormalized[:,2,i])
            pixel_coordinates[:,1,i] = np.divide(pixel_coordinates_unnormalized[:,1,i], pixel_coordinates_unnormalized[:,2,i])

        return pixel_coordinates

    def draw_line_in_I_from_points_in_I(self, image_path, points_in_I, linestyle='o'):

        if points_in_I.shape[1] > 3:
            points_in_I = np.transpose(points_in_I)

        self.perspective_projection.load_image(image_path)

        plt.plot(points_in_I[:,0], points_in_I[:,1], linestyle)

        self.perspective_projection.display_image()

        return