import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from numpy.core.numeric import full

class EightPoint:
    def __init__(self, K1, K2):

        self.K1 = K1
        self.K2 = K2

        return

    def fundamental_eight_point(self, p1, p2):
        """
        The 8-point algorithm for the estimation of the fundamental matrix, F, with a posteriori enforcement of the singularity constraint, det(F) = 0. Does not include data normalization.
        
        Reference: "Multiple View Geometry" (Hartley & Zisserman, 2000), Section 10.1, page 262.

        Input: Point correspondences
        p1(N,3): homogeneous coordinates of 2D points in image 1
        p2(N,3): homogeneous coordinates of 2D points in image 2

        Output:
        F(3,3): fundamental matrix
        """
        num_points = p1.shape[0]
        if p1.shape[1] > 2 or p2.shape[1] > 2:
            assert "Must provide 2D points!"
        if p1.shape[0] < 8 or p2.shape[0] < 8:
            assert "Must provide at least eight 2D points!"

        # Compute the measurement matrix A of the linear homogeneous system whose solution is the vector representing the fundamental matrix.
        Q = np.zeros(num_points, 9)
        for i in range(num_points):
            Q[i,:] = np.kron(p1[i,:], p2[i,:]).transpose()
        
        # ''Solve'' the linear homogeneous system of equations, A*f = 0
        # The correspondences x1, x2 are exact <=> rank(A) = 8 -> there exists an exact solution
        # If measurements are noisy, then rank(A) = 9 -> there is no exact solution, seek a least-squares solution
        _, _, Vt = np.linalg.svd(Q, full_matrices=True)
        F = np.reshape(Vt[-1,:], (3,3), order='F')

        # Enforce det(F) = 0 by projecting F onto the set of 3x3 singular matrices
        U, S, Vt = np.linalg.svd(F)
        S[-1,-1] = 0
        F = np.matmul(U, np.matmul(S, Vt))

        return F

    def estimate_essential_matrix(self, p1, p2):

        return

class Triangulation:
    def __init__(self):

        return

    @staticmethod
    def linear_triangulation(p1, p2, M1, M2):
        """
        LINEARTRIANGULATION Linear trianguation

        Input:
        p1(N,3) : homogeneous coordinates of points in image 1
        p2(N,3) : homogeneous coordinates of points in image 2
        M1(3,4) : projection matrix corresponding to first image
        M2(3,4) : projection matrix corresponding to second image

        Output:
        P(N,4) : homogeneous coordinates of 3-D points
        """
        
        num_points = p1.shape[0]
        P = np.zeros((num_points, 4))

        for j in range(num_points):
            # Build matrix of linear homogeneous systems of equations
            A1 = np.matmul(\
                Triangulation.cross_2_matrix(p1[j,:]),
                M1)

            A2 = np.matmul(\
                Triangulation.cross_2_matrix(p2[j,:]),
                M2)
            
            A = np.concatenate((A1, A2), axis=0)

            # Solve linear system of equations
            _, _, Vt = np.linalg.svd(A, full_matrices=True)
            P[j,:] = Vt[-1,:]

        P = np.divide(P, np.tile(P[:,3,np.newaxis], (1,4)))

        return P

    @staticmethod
    def cross_2_matrix(x):
        """
        CROSS_2_MATRIX: Computes antisymmetric matrix corresponding to a 3-component vector such that
        M*y = cross(x,y) for all 3-component vectors, y.

        Input:
        x(3,1) : vector

        Output:
        M(3,3) : antisymmetric matrix
        """

        M = np.array([\
            [0, -x[2], x[1]],\
            [x[2], 0, -x[0]],\
            [-x[1], x[0], 0]\
            ])

        return M