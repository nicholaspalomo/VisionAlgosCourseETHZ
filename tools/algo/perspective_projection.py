from operator import matmul
from matplotlib import image
import numpy as np
import cv2 # OpenCV
import math
from matplotlib import pyplot as plt
import os

from ..utils.process_text_file import ProcessTextFile

class PerspectiveProjection:
    def __init__(self, camera_K_matrix, camera_D_matrix):
        self.camera_K_matrix_ = camera_K_matrix
        self.camera_D_matrix_ = camera_D_matrix
        self.grayscale_image_ = []

    def load_image(self, image_fname, display=False):
        color_image = cv2.imread(image_fname)
        self.grayscale_image_ = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        if display:
            self.display_image()

        return

    def display_image(self):
        plt.imshow(self.grayscale_image_)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show(block=False)

        return

    def scatter(self, points):

        plt.scatter(points[:,0], points[:,1])

        return

    def animation_with_drawing_in_I(self, image_path, points_in_W, camera_poses, linestyle='o', decimation=0.001, image_distortion=False):

        i = 0
        for frame in sorted(os.listdir(image_path)):
            self.draw_line_in_I(image_path + "/" + frame, points_in_W, camera_poses[i,:], linestyle=linestyle, image_distortion=image_distortion)
            plt.pause(decimation)
            plt.clf()
            i += 1

        return

    def draw_line_in_I(self, image_path, points_in_W, camera_pose, linestyle='o', image_distortion=False):
        if points_in_W.shape[1] > 3:
            points_in_W = np.transpose(points_in_W)

        self.load_image(image_path)

        plot_points_in_I = np.zeros((points_in_W.shape[0], 2))
        i = 0
        for j in range(points_in_W.shape[0]):
            points_in_I, _ = self.project_W_to_I(camera_pose, np.reshape(points_in_W[j,:], (3, 1)), image_distortion=image_distortion)
            plot_points_in_I[i, :] = np.transpose(points_in_I)
            i += 1

        plt.plot(plot_points_in_I[:,0], plot_points_in_I[:,1], linestyle)

        self.display_image()

        return

    def project_W_to_I(self, camera_pose, point_in_W, image_distortion=False):

        point_in_C = self.project_W_to_C(camera_pose, point_in_W, image_distortion=image_distortion)

        pixel_coordinates, lam = self.project_C_to_I(point_in_C, image_distortion=image_distortion)

        return pixel_coordinates, lam # [u, v], lambda

    def project_W_to_C(self, camera_pose, point_in_W, image_distortion=False):
        _, _, transform_mat_W_to_C = PerspectiveProjection.get_transform_mat(camera_pose)

        point_in_W = np.reshape(point_in_W, (3,1))
        point_in_W = np.concatenate((point_in_W, np.ones((1,1))), axis=0)

        point_in_C = np.matmul(transform_mat_W_to_C, point_in_W)

        if image_distortion:
            point_in_C_normalized = point_in_C / point_in_C[2]

            point_in_C = self.distort_points(point_in_C_normalized)

        return point_in_C

    def distort_points(self, point_in_C_normalized):

        k1, k2 = self.camera_D_matrix_[0], self.camera_D_matrix_[1]
        r = np.linalg.norm(point_in_C_normalized[:2])

        # Apply lens distortion model
        point_in_C_ = point_in_C_normalized
        point_in_C_[:2] = (1 + k1 * r**2 + k2 * r**4) * point_in_C_normalized[:2]

        return point_in_C_

    def project_C_to_I(self, point_in_C, image_distortion=False):

        projected_points = np.matmul(point_in_C, self.camera_K_matrix_.transpose())
        projected_points /= np.tile(projected_points[:,-1, np.newaxis], (1,3))
        if image_distortion:
            projected_points = self.distort_points(projected_points)

        return projected_points[:,:2], projected_points[:,2] # [u, v], lambda

    def undistort_image(self, image_path, bilinear=True):
        self.load_image(image_path)

        K_inv = np.linalg.inv(self.camera_K_matrix_)
        k1, k2 = self.camera_D_matrix_[0], self.camera_D_matrix_[1]

        unnormalized_pixel_coords = self.meshgrid(\
            self.grayscale_image_.shape[1],\
            self.grayscale_image_.shape[0])

        normalized_image_coords = self.pixel_coords_2_normalized(unnormalized_pixel_coords)

        r = np.linalg.norm(\
            normalized_image_coords[:, :, :2],\
            axis=2)

        distortion_factor = np.repeat(\
            np.expand_dims(1 + k1 * r**2 + k2 * r**4, 2), 2,\
            axis=2)

        distorted_normalized_coordinates = normalized_image_coords.copy()
        distorted_normalized_coordinates[:, :, :2] = np.multiply(\
            distortion_factor,\
            normalized_image_coords[:, :, :2])
        
        distorted_pixel_coords = self.normalized_2_pixel_coords(distorted_normalized_coordinates)
        distorted_pixel_coords = self.clip_pixel_coords(distorted_pixel_coords)

        image_shape = self.grayscale_image_.shape
        # if bilinear:
        #     img_BR = self.grayscale_image_[\
        #         np.clip(np.floor(distorted_pixel_coords[:,:,1] + 1).astype(int).reshape((-1,1)), 0, image_shape[0]-1),\
        #         np.clip(np.floor(distorted_pixel_coords[:,:,0] + 1).astype(int).reshape((-1,1)), 0, image_shape[1]-1)].reshape(image_shape)
            
        #     img_BL = self.grayscale_image_[\
        #         np.clip(np.floor(distorted_pixel_coords[:,:,1] + 1).astype(int).reshape((-1,1)), 0, image_shape[0]-1),\
        #         np.clip(np.ceil(distorted_pixel_coords[:,:,0] - 1).astype(int).reshape((-1,1)), 0, image_shape[1]-1)].reshape(image_shape)

        #     img_TL = self.grayscale_image_[\
        #         np.clip(np.ceil(distorted_pixel_coords[:,:,1] - 1).astype(int).reshape((-1,1)), 0, image_shape[0]-1),\
        #         np.clip(np.ceil(distorted_pixel_coords[:,:,0] - 1).astype(int).reshape((-1,1)), 0, image_shape[1]-1)].reshape(image_shape)

        #     img_TR = self.grayscale_image_[\
        #         np.clip(np.ceil(distorted_pixel_coords[:,:,1] - 1).astype(int).reshape((-1,1)), 0, image_shape[0]-1),\
        #         np.clip(np.floor(distorted_pixel_coords[:,:,0] + 1).astype(int).reshape((-1,1)), 0, image_shape[1]-1)].reshape(image_shape)

        #     self.grayscale_image_ = 0.25 * (img_BR + img_BL + img_TL + img_TR)

        # else:

        self.grayscale_image_ = self.grayscale_image_\
            [distorted_pixel_coords[:,:,1].astype(int).reshape((-1,1)),\
            distorted_pixel_coords[:,:,0].astype(int).reshape((-1,1))].reshape(image_shape)

        self.display_image()

        return

    def normalized_2_pixel_coords(self, normalized_coords):

        return np.matmul(normalized_coords,\
            self.camera_K_matrix_.transpose())

    def pixel_coords_2_normalized(self, pixel_coords):

        return np.matmul(\
            pixel_coords,\
            np.linalg.inv(self.camera_K_matrix_).transpose())

    def clip_pixel_coords(self, coords):

        coords[:,:,0] = np.clip(\
            coords[:,:,0], 0, self.grayscale_image_.shape[1]-1).copy()
        coords[:,:,1] = np.clip(\
            coords[:,:,1], 0, self.grayscale_image_.shape[0]-1).copy()

        return coords


    def meshgrid(self, x_dim, y_dim):
        mesh_x, mesh_y = np.meshgrid(np.linspace(0, x_dim-1, x_dim), np.linspace(0, y_dim-1, y_dim))

        return np.dstack((mesh_x, mesh_y, np.ones((mesh_x.shape))))

    @staticmethod
    def get_transform_mat(camera_pose):
        rot_mat_W_to_C = PerspectiveProjection.angle_axis_2_rot_mat(camera_pose[:3])
        t_pos_in_W = np.reshape(camera_pose[3:], (3,1))

        return rot_mat_W_to_C, t_pos_in_W, np.concatenate((rot_mat_W_to_C, t_pos_in_W), axis=1)

    @staticmethod
    def angle_axis_2_rot_mat(angle_axis):
        eye3 = np.eye(3)
        theta = np.linalg.norm(angle_axis)
        k = angle_axis / theta
        
        k_skew_symmetrix = PerspectiveProjection.skew_symmetric_3(k)

        # Apply Rodrigues formula to get the unnormalized rotation matrix from the angle axis representation
        rot_mat = eye3 + math.sin(theta) * k_skew_symmetrix + (1 - math.cos(theta)) * k_skew_symmetrix * k_skew_symmetrix

        return PerspectiveProjection.orthonormal_mat(rot_mat)

    @staticmethod
    def orthonormal_mat(mat):
        # Perform SVD on rotation matrix to make the rows and columns orthonormal
        U, _, V_transpose = np.linalg.svd(mat, full_matrices=True)
        return np.matmul(U, V_transpose)

    @staticmethod
    def skew_symmetric_3(vec):
        skew_symmetrix = np.zeros((3,3))
        skew_symmetrix[0, 1] = -vec[2]
        skew_symmetrix[0, 2] = vec[1]
        skew_symmetrix[1, 2] = -vec[0]
        skew_symmetrix -= np.transpose(skew_symmetrix)

        return skew_symmetrix