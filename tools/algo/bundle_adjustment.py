import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from scipy.linalg import expm, logm
from scipy import optimize

class BundleAdjustment:
    def __init__(self):

        return

    def align_estimate_to_ground_truth(self, pp_G_C, p_V_C):
        """
        Returns the points of the estimated trajectory p_V_C transformed into the ground truth frame, G. The similarity transform Sim_G_V is to be chosen such that it results in the lowest error between the aligned trajectory points, p_G_C, and the points of the ground truth trajectory, pp_G_C. All matrices are 3xN.
        """
        
        # Initial guess is identity
        twist_guess = BundleAdjustment.homog_matrix_2_twist(np.eye(4))
        scale_guess = 1

        x = np.squeeze(np.concatenate((twist_guess, np.array([[scale_guess]])), axis=0))
        # Using an external error function for nicer code. Binding pp_G_C and p_V_C by casting the function as a function of the hidden state only.
        # err_fun = BundleAdjustment.align_error(x, pp_G_C, p_V_C) # uncomment this line to evaluate error function (debugging)

        x_optim = optimize.least_squares(BundleAdjustment.align_error, x, args=(pp_G_C, p_V_C), verbose=2)

        T_G_V = BundleAdjustment.twist_2_homog_matrix(x_optim["x"][:6])
        scale_G_V = x_optim["x"][6]

        num_frames = p_V_C.shape[0]

        return np.transpose(scale_G_V * np.matmul(T_G_V[:3, :3], p_V_C.transpose()) + np.tile(T_G_V[:3, 3], (num_frames, 1)).transpose()) # p_G_C

    def plot_map(self, hidden_state, observations, ax_range, ax=plt.gca()):

        num_frames = int(observations[0])
        T_W_frames = np.reshape(hidden_state[:6*num_frames], (6,-1), order='F')
        p_W_landmarks = np.reshape(hidden_state[6*num_frames:], (3,-1))

        p_W_frames = np.zeros((3, num_frames))
        for i in range(num_frames):
            T_W_frame = BundleAdjustment.twist_2_homog_matrix(T_W_frames[:,i])
            p_W_frames[:,i] = T_W_frame[:3, -1]

        ax.scatter(p_W_landmarks[2, :], -p_W_landmarks[0, :], s=2)
        ax.scatter(p_W_frames[2, :], -p_W_frames[0, :], s=2, marker='x', color='r')

        ax.axis('equal')
        ax.axis(ax_range)

        return

    @staticmethod
    def align_error(x, pp_G_C, p_V_C):
        """
        Given x, which encodes the similarity transform Sim_G_V as a concatenation of twist and scale, return the error pp_G_C - p_G_C (p_G_C = Sim_G_V * p_V_C) as a single column vector.
        """

        T_G_V = BundleAdjustment.twist_2_homog_matrix(x[:6])
        scale_G_V = x[6]

        num_frames = p_V_C.shape[0]
        p_G_C = np.transpose( scale_G_V * np.matmul(T_G_V[:3, :3], p_V_C.transpose()) + np.tile(T_G_V[:3, 3], (num_frames, 1)).transpose() )

        errors = pp_G_C - p_G_C

        return np.reshape(errors, (-1,), order='F')

    def crop_problem(self, hidden_state, observations, ground_truth, cropped_num_frames):
        """
        Determine which landmarks to keep; assuming landmark indices increase with frame indices

        Inputs:
        hidden_state
        observations
        ground_truth
        cropped_num_frames

        Outputs:
        cropped_hidden_state
        cropped_observations
        cropped_ground_truth

        """

        if ground_truth.shape[0] == 3 and ground_truth.shape[1] > ground_truth.shape[0]:
            ground_truth_ = np.transpose(ground_truth.copy())
        else:
            ground_truth_ = ground_truth.copy()

        num_frames = observations[0]
        assert cropped_num_frames < num_frames, "cropped_num_frames should be less than num_frames"

        observation_i = 2
        cropped_num_landmarks = 0
        for i in range(cropped_num_frames):
            num_observations = observations[observation_i]
            if i == cropped_num_frames-1:
                cropped_num_landmarks = np.max(observations[int(observation_i + 1 + num_observations*2):int(observation_i + num_observations*3)])
            
            observation_i += int(num_observations*3 + 1)

        cropped_hidden_state = np.concatenate(( \
            hidden_state[:6*cropped_num_frames, np.newaxis], \
            hidden_state[int(6*num_frames):int(6*num_frames + 3*cropped_num_landmarks), np.newaxis]), axis = 0) # (nico) : should first index here be 6*num_frames + 1?

        cropped_observations = np.concatenate(( \
            np.array([[cropped_num_frames], [cropped_num_landmarks]]), \
            observations[2:observation_i-1, np.newaxis]), axis=0)

        cropped_ground_truth = ground_truth_[0:cropped_num_frames,:]

        return np.squeeze(cropped_hidden_state), np.squeeze(cropped_observations), cropped_ground_truth

    @staticmethod
    def twist_2_homog_matrix(twist):
        """
        Convert twist coordinates to 4x4 homogeneous matrix

        Input:
        twist - (6,1) twist coordinates. Stack linear and angular parts [v; w]

        Output:
        H - (4,4) Euclidean transformation matrix (rigid body motion)

        """

        v = np.squeeze(twist[:3]) # linear part
        w = twist[3:]

        se_matrix = np.concatenate( \
            (BundleAdjustment.cross_2_matrix(np.squeeze(w)), v[:,np.newaxis]), axis=1)
        se_matrix = np.concatenate( \
            (se_matrix, np.array([0., 0., 0., 0.])[np.newaxis, :]), axis=0)

        return expm(se_matrix)

    @staticmethod
    def homog_matrix_2_twist(H):
        """
        Convert 4x4 homogeneous matrix to twist coordinates

        Input:
        H : 4x4 Euclidean transformation matrix (rigid body motion)

        Output:
        twist : 6x1 Twist coordinates. Stack linear and angular parts [v;w]

        Observe that the same H might be represented by different twist vectors.
        Here, twist[3:] is a rotation vector with norm in [0, pi]
        """

        se_matrix = logm(H)

        # Careful for rotations of pi; the top 3x3 submatrix of the returned se_matrix by logm is not skew-symmetric (bad).

        v = se_matrix[:3, 3, np.newaxis]

        w = BundleAdjustment.matrix_2_cross(se_matrix[:3,:3])

        twist = np.concatenate((v, w), axis=0)

        return twist

    @staticmethod
    def cross_2_matrix(x):
        """
        Computes antisymmetric matrix, M, corresponding to a 3D vector, x, such that M*y = cross(x,y) for all 3D vectors y.

        Input:
        x - (3,1) vector

        Output:
        M - (3,3) antisymmetric matrix
        """

        return np.array([ \
            [0.,     -x[2], x[1]], \
            [x[2],  0.,     -x[0]], \
            [-x[1], x[0],  0.]])

    @staticmethod
    def matrix_2_cross(M):
        """
        Compute 3D vector, x, corresponding to an antisymmetric matrix, M, such that M*y = cross(x,y) for all 3D vectors y.

        Input:
        M - (3,3) antisymmetric matrix

        Output:
        x - (3,1) column vector
        """

        return np.array([[-M[1,2]], [M[0,2]], [-M[0,1]]])