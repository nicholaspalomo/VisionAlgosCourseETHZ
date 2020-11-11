from os import stat
import numpy as np
import cv2 # OpenCV
import math

from numpy.linalg.linalg import qr

class Rotations:
    def __init__(self):

        return

    @staticmethod
    def rot_mat_2_quat(R):
        """
        Convert a rotation from rotation matrix to quaternion representation. Assumes that the columns of the rotation matrix are orthonomal!
        """

        i, j, k = 0, 1, 2
        if R[1,1] > R[0,1]:
            i, j, k = 1, 2, 0

        if R[2,2] > R[i,i]:
            i, j, k = 2, 0, 1

        t = R[i,i] - (R[j,j] + R[k,k]) + 1
        q = np.array([0, 0, 0, 0])
        q[0] = R[k,j] - R[j,k]
        q[i+1] = t
        q[j+1] = R[i,j] + R[j,i]
        q[k+1] = R[k,i] + R[i,k]
        q = np.multiply(q, 0.5 / math.sqrt(t))

        return q

    @staticmethod
    def quat_2_rot_mat(q):
        """
        Convert a rotation from rotation matrix to quaternion representation.
        """

        s = np.linalg.norm(q) # s = 1 if the quaternion has unit length

        R = np.zeros((3,3))
        R[0,0] = 1 - 2 * s * (q[2]**2 + q[3]**2)
        R[0,1] = 2 * s * (q[1]*q[2] - q[3]*q[0])
        R[0,2] = 2 * s * (q[1]*q[3] + q[2]*q[0])
        R[1,0] = 2 * s * (q[1]*q[2] + q[3]*q[0])
        R[1,1] = 1 - 2 * s * (q[1]**2 + q[3]**2)
        R[1,2] = 2 * s * (q[2]*q[3] - q[1]*q[0])
        R[2,0] = 2 * s * (q[1]*q[3] - q[2]*q[0])
        R[2,1] = 2 * s * (q[2]*q[3] + q[1]*q[0])
        R[2,2] = 1 - 2 * s * (q[1]**2 + q[2]**2)

        R = Rotations.orthonormal_mat(R)

        return R

    @staticmethod
    def rot_mat_2_angle_axis(rot_mat):
        theta = math.acos((np.trace(rot_mat) - 1) / 2)

        return 1 / (2 * math.sin(theta)) * np.array([rot_mat[2,1] - rot_mat[1,2], rot_mat[0,2] - rot_mat[2,0], rot_mat[1,0] - rot_mat[0,1]]).reshape((3,1))

    @staticmethod
    def orthonormal_mat(mat):
        # Perform SVD on rotation matrix to make the rows and columns orthonormal
        U, _, V_transpose = np.linalg.svd(mat, full_matrices=True)
        return np.matmul(U, V_transpose)