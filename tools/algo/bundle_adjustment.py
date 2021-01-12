import numpy as np
import cv2
import math
from scipy.linalg import expm

class BundleAdjustment:
    def __init__(self):

        return

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
            hidden_state[np.newaxis, 0:6*cropped_num_frames], \
            hidden_state[np.newaxis, int(6*num_frames):int(6*num_frames + 3*cropped_num_landmarks)]), axis = 0)

        cropped_observations = np.concatenate(( \
            np.array([[cropped_num_frames], [cropped_num_landmarks]]), \
            observations[2:observation_i-1]), axis=0)

        cropped_ground_truth = ground_truth[:, 0:cropped_num_frames]

        return cropped_hidden_state, cropped_observations, cropped_ground_truth

    @staticmethod
    def twist_2_homog_matrix(twist):
        """
        Convert twist coordinates to 4x4 homogeneous matrix

        Input:
        twist - (6,1) twist coordinates. Stack linear and angular parts [v; w]

        Output:
        H - (4,4) Euclidean transformation matrix (rigid body motion)

        """

        v = twist[:3] # linear part
        w = twist[3:]

        se_matrix = np.concatenate( \
            (BundleAdjustment.cross_2_matrix(w), v[:,np.newaxis]), axis=1)
        se_matrix = np.concatenate( \
            (se_matrix, np.array([0., 0., 0., 0.])[np.newaxis, :]), axis=0)

        return expm(se_matrix)

    @staticmethod
    def cross_2_matrix(x):
        """
        Computes antisymmetric matrix, M, corresponding to a 3D vector, x, such that M*y = cross(x,y) for all 3D vectors y

        Input:
        x - (3,1) vector

        Output:
        M - (3,3) antisymmetric matrix
        """

        return np.array([ \
            [0.,     -x[2], x[1]], \
            [x[2],  0.,     -x[0]], \
            [-x[1], x[0],  0.]])