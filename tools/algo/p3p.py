import cv2
import math
import numpy as np
import cmath
from numpy.linalg.linalg import norm

# Reference: ''A Novel Parametrization of the P3P-Problem for a Direct Computation of Absolute Camera Position and Orientation''
class P3P:
    def __init__(self):

        return

    def p3p(self, worldPoints, imageVectors):
        """
        Compute the camera pose using 2D-to-3D correspondences

        Inputs:
        worldPoints - 3x3 matrix with corresponding 3D world points (each column is a point)
        imageVectors - 3x3 matrix with UNITARY feature vectors (each column is a vector)

        Outputs:
        poses - 3x16 matrix containing multiple solutions of the form:
            [3x1 position (solution1), 3x3 orientation (solution1), 3x1 position (solution2), 3x3 orientation (solution2)]
            Obtained orientation matrices are defined as transforming points from the CAMERA TO THE WORLD FRAME, T_WC
        """

        # Initialization of the solution matrix, extraction of world points and feature vectors

        poses = np.zeros((3, 4*4))

        P1 = worldPoints[:,0]
        P2 = worldPoints[:,1]
        P3 = worldPoints[:,2]

        # Verification that world points are not colinear

        vector1 = P2 - P1
        vector2 = P3 - P1

        if np.linalg.norm(np.cross(vector1, vector2)) < 1e-5:
            return poses

        # Creation of intermediate camera frame

        f1 = imageVectors[:, 0]
        f2 = imageVectors[:, 1]
        f3 = imageVectors[:, 2]

        e1 = f1
        e3 = np.cross(f1, f2)
        norm_e3 = np.sum(np.matmul(e3, e3))**0.5
        e3 /= norm_e3
        e2 = np.cross(e3, e1)

        T = np.concatenate((e1[:,np.newaxis].transpose(), e2[:,np.newaxis].transpose(), e3[:,np.newaxis].transpose()), axis=0)

        f3 = np.matmul(T, f3[:,np.newaxis])

        # Ensure that f3[2] > 0 such that theta is in range [0, pi]

        if f3[2] > 0:

            f1 = imageVectors[:, 1]
            f2 = imageVectors[:, 0]
            f3 = imageVectors[:, 2]

            e1 = f1
            e3 = np.cross(f1, f2)
            norm_e3 = np.sum(np.matmul(e3, e3))**0.5
            e3 /= norm_e3
            e2 = np.cross(e3, e1)

            T = np.concatenate((e1[:,np.newaxis].transpose(), e2[:,np.newaxis].transpose(), e3[:,np.newaxis].transpose()), axis=0)

            f3 = np.matmul(T, f3[:,np.newaxis])

            P1 = worldPoints[:, 1]
            P2 = worldPoints[:, 0]
            P3 = worldPoints[:, 2]

        # Creation of intermediate world frame

        n1 = P2 - P1
        norm_n1 = np.sum(np.matmul(n1, n1))**0.5
        n1 /= norm_n1
        n3 = np.cross(n1, P3 - P1)
        norm_n3 = np.sum(np.matmul(n3, n3))**0.5
        n3 /= norm_n3
        n2 = np.cross(n3, n1)

        N = np.concatenate((n1[:,np.newaxis].transpose(), n2[:,np.newaxis].transpose(), n3[:,np.newaxis].transpose()), axis=0)

        # Extraction of known parameters
        
        P3 = np.matmul(N, (P3 - P1)[:,np.newaxis])

        d_12 = np.sum((P2 - P1)**2)**0.5
        f_1 = f3[0,0] / f3[2,0]
        f_2 = f3[1,0] / f3[2,0]
        p_1 = P3[0,0]
        p_2 = P3[1,0]

        cos_beta = np.sum(f1*f2)
        b = 1/(1 - cos_beta**2) - 1

        if cos_beta < 0:
            b = -cmath.sqrt(b)
        else:
            b = cmath.sqrt(b)

        # Definition of temporary variables for avoiding multiple computation

        f_1_pw2 = f_1**2
        f_2_pw2 = f_2**2
        p_1_pw2 = p_1**2
        p_1_pw3 = p_1_pw2 * p_1
        p_1_pw4 = p_1_pw3 * p_1
        p_2_pw2 = p_2**2
        p_2_pw3 = p_2_pw2 * p_2
        p_2_pw4 = p_2_pw3 * p_2
        d_12_pw2 = d_12**2
        b_pw2 = b**2

        # Computation of fators of 4th degree polynomial

        factor_4 = -f_2_pw2*p_2_pw4 \
                -p_2_pw4*f_1_pw2 \
                -p_2_pw4

        factor_3 = 2*p_2_pw3*d_12*b \
                +2*f_2_pw2*p_2_pw3*d_12*b \
                -2*f_2*p_2_pw3*f_1*d_12

        factor_2 = -f_2_pw2*p_2_pw2*p_1_pw2 \
                -f_2_pw2*p_2_pw2*d_12_pw2*b_pw2 \
                -f_2_pw2*p_2_pw2*d_12_pw2 \
                +f_2_pw2*p_2_pw4 \
                +p_2_pw4*f_1_pw2 \
                +2*p_1*p_2_pw2*d_12 \
                +2*f_1*f_2*p_1*p_2_pw2*d_12*b \
                -p_2_pw2*p_1_pw2*f_1_pw2 \
                +2*p_1*p_2_pw2*f_2_pw2*d_12 \
                -p_2_pw2*d_12_pw2*b_pw2 \
                -2*p_1_pw2*p_2_pw2

        factor_1 = 2*p_1_pw2*p_2*d_12*b \
                +2*f_2*p_2_pw3*f_1*d_12 \
                -2*f_2_pw2*p_2_pw3*d_12*b \
                -2*p_1*p_2*d_12_pw2*b

        factor_0 = -2*f_2*p_2_pw2*f_1*p_1*d_12*b \
                +f_2_pw2*p_2_pw2*d_12_pw2 \
                +2*p_1_pw3*d_12 \
                -p_1_pw2*d_12_pw2 \
                +f_2_pw2*p_2_pw2*p_1_pw2 \
                -p_1_pw4 \
                -2*f_2_pw2*p_2_pw2*p_1*d_12 \
                +p_2_pw2*f_1_pw2*p_1_pw2 \
                +f_2_pw2*p_2_pw2*d_12_pw2*b_pw2

        # Computation of roots

        x = self.solveQuartic((factor_4, factor_3, factor_2, factor_1, factor_0))

        # Backsubstitution of each solution

        for i in range(4):

            cot_alpha = (-f_1*p_1/f_2-np.real(x[i])*p_2+d_12*b)/(-f_1*np.real(x[i])*p_2/f_2+p_1-d_12)
            
            cos_theta = np.real(x[i])
            sin_theta = cmath.sqrt(1-np.real(x[i])**2)
            sin_alpha = cmath.sqrt(1/(cot_alpha**2+1))
            cos_alpha = cmath.sqrt(1-sin_alpha**2)
            
            if cot_alpha < 0:
                cos_alpha = -cos_alpha

            C = np.array((d_12*cos_alpha*(sin_alpha*b+cos_alpha), \
                cos_theta*d_12*sin_alpha*(sin_alpha*b+cos_alpha), \
                sin_theta*d_12*sin_alpha*(sin_alpha*b+cos_alpha)))[:,np.newaxis]
            
            C = P1[:,np.newaxis] + np.matmul(np.transpose(N), C)

            R = np.array([[-cos_alpha, -sin_alpha*cos_theta, -sin_alpha*sin_theta], \
                [sin_alpha, -cos_alpha*cos_theta, -cos_alpha*sin_theta], \
                [0, -sin_theta, cos_theta]])
            
            R = np.matmul(np.matmul(np.transpose(N), np.transpose(R)), T)
            
            poses[:, 4*i] = C[:,0]
            poses[:, 4*i+1:4*i+4] = R

        return poses

    def solveQuartic(self, factors):

        A = factors[0]
        B = factors[1]
        C = factors[2]
        D = factors[3]
        E = factors[4]
        
        A_pw2 = A*A
        B_pw2 = B*B
        A_pw3 = A_pw2*A
        B_pw3 = B_pw2*B
        A_pw4 = A_pw3*A
        B_pw4 = B_pw3*B
        
        alpha = -3*B_pw2/(8*A_pw2)+C/A
        beta = B_pw3/(8*A_pw3)-B*C/(2*A_pw2)+D/A
        gamma = -3*B_pw4/(256*A_pw4)+B_pw2*C/(16*A_pw3)-B*D/(4*A_pw2)+E/A
        
        alpha_pw2 = alpha*alpha
        alpha_pw3 = alpha_pw2*alpha
        
        P = -alpha_pw2/12-gamma
        Q = -alpha_pw3/108+alpha*gamma/3-beta**2/8
        R = -Q/2+cmath.sqrt(Q**2/4+P**3/27)
        U = R**(1/3)
        
        if U == 0:
            y = -5*alpha/6-Q**(1/3)
        else:
            y = -5*alpha/6-P/(3*U)+U
        
        w = cmath.sqrt(alpha+2*y)
        
        roots = np.zeros((4,))
        roots[0] = -B/(4*A) + 0.5*(w+cmath.sqrt(-(3*alpha+2*y+2*beta/w)))
        roots[1] = -B/(4*A) + 0.5*(w-cmath.sqrt(-(3*alpha+2*y+2*beta/w)))
        roots[2] = -B/(4*A) + 0.5*(-w+cmath.sqrt(-(3*alpha+2*y-2*beta/w)))
        roots[3] = -B/(4*A) + 0.5*(-w-cmath.sqrt(-(3*alpha+2*y-2*beta/w)))

        return roots