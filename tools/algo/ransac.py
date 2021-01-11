import numpy as np
from numpy import matlib
import cv2 # OpenCV
import math
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import Inf

from tools.algo.harris import Harris
from tools.algo.dlt import DLT
from tools.algo.p3p import P3P

class RANSAC:
    def __init__(self, num_iterations=100):
        self.num_iterations = num_iterations
        self.pixel_tolerance = 10
        self.k = 3 # for PnP, DLT methods

        self.detector = None # e.g. Harris, SIFT, etc. Some algorithm to detect keypoints

        self.database_keypoints = None
        self.query_keypoints = None
        self.matched_query_keypoints = None
        self.all_matches = None
        self.inlier_mask = None

        return

    def parabola_ransac(self, data, max_noise, rerun_on_inliers=True):
        """
        Inputs:
        data - 2 x N matrix with the data points given column-wise

        best_guess_history - 3 x num_iterations ith the polynomial coefficients from polyfit of the BEST GUESS SO FAR at each iteration columnwise

        max_num_inliers_history - 1 x num_iterations with the inlier count of the BEST GUESS SO FAR at each iteration
        """

        best_guess_history = np.zeros((3, self.num_iterations))
        max_num_inliers_history = np.zeros((1, self.num_iterations))

        best_guess = np.zeros((3, 1)) # coefficients of polynomial fit to data
        max_num_inliers = 0
        num_samples = 3

        for i in range(self.num_iterations):
            # Model based on 3 samples
            samples = data[:, np.random.choice(data.shape[1], num_samples, replace=False)]
            guess = np.polyfit(samples[0, :], samples[1, :], 2) # fit a polynomial to the random sample of x-y points

            # Evaluate number of inliers
            errors = np.abs(np.polyval(guess, data[0, :]) - data[1, :])
            inliers = errors <= max_noise + 1e-5
            num_inliers = np.count_nonzero(inliers)

            # Determine if the current guess is the best so far
            if num_inliers > max_num_inliers:
                if rerun_on_inliers:
                    guess = np.polyfit(data[0, inliers], data[1, inliers], 2)

                best_guess = np.transpose(guess)
                max_num_inliers = num_inliers

            best_guess_history[:, i] = best_guess
            max_num_inliers_history[0, i] = max_num_inliers

        return best_guess_history, max_num_inliers_history

    def detect_localize_landmarks_ransac(self, p_W_landmarks, K, harris_params, database_img, database_keypoints, query_img_path, tweaked_for_more=True, use_p3p=False):

        self.detector = Harris(harris_params)

        query_keypoints, query_descriptors = self.detector.get_keypoints_descriptors_from_image(query_img_path)

        database_descriptors = self.detector.get_keypoint_descriptors(database_keypoints, database_img)

        all_matches = self.detector.match_descriptors(query_descriptors, database_descriptors)

        # Drop unmatched keypoints and get 3D landmarks for the matched ones
        matched_query_keypoints = query_keypoints[all_matches > 0, :]
        corresponding_matches = all_matches[all_matches > 0].astype(np.int)
        corresponding_landmarks = p_W_landmarks[corresponding_matches, :]

        self.database_keypoints = database_keypoints
        self.query_keypoints = query_keypoints # query keypoints - potential keypoints detected by e.g. Harris corner detector, SIFT, etc...
        self.matched_query_keypoints = query_keypoints[all_matches > 0, :]
        self.all_matches = all_matches

        return self.ransac_localization(matched_query_keypoints, corresponding_landmarks, K, use_p3p=use_p3p, tweaked_for_more=tweaked_for_more)

    def ransac_localization(self,  matched_query_keypoints, corresponding_landmarks, K, use_p3p=False, tweaked_for_more=True, adaptive=True):
        """
        Inputs:
        matched_query_keypoints - keypoints should be 2 x num_keypoints. All matches should be 1 x num_keypoints and correspond to the output from the match_descriptors method of Harris class
        corresponding_landmarks - matched 3D landmarks
        K - camera matrix
        use_p3p
        tweaked_for_more
        adaptive - whether or not to use RANSAC adaptively

        Outputs:
        R_C_W - rotation matrix from world frame to camera
        t_C_W - translation from world frame to camera
        best_inlier_mask
        max_num_inliers_history
        num_iteration_history

        Notes:
        best_inlier_mask should be 1 x num_matched (!!!) and contain, only for the matched keypoints (!!!), 0 if the match is an outlier; 1 otherwise
        """

        # 1. Find keypoints in query image
        if use_p3p:
            if tweaked_for_more:
                self.num_iterations = 1000
            else:
                self.num_iterations = 200
            self.pixel_tolerance = 10
            self.k = 3 # for P3P method
        else:
            self.num_iterations = 2000
            self.pixel_tolerance = 10
            self.k = 6

        if adaptive:
            self.num_iterations = Inf

        # Initialize RANSAC
        best_inlier_mask = np.zeros((1, matched_query_keypoints.shape[0]))

        # (row, col) to (u, v)
        matched_query_keypoints = np.fliplr(matched_query_keypoints)

        max_num_inliers_history = []
        num_iteration_history = []
        max_num_inliers = 0

        # RANSAC
        i = 1

        dlt = DLT(K, [0., 0.])
        p3p = P3P()
        K_inv = np.linalg.inv(K)

        M_C_W_guess = np.zeros((3,4,4))
        while self.num_iterations > i:
            # Model from k samples (DLT or P3P)
            idx = np.random.choice(corresponding_landmarks.shape[0], self.k, replace=False)
            landmark_sample = corresponding_landmarks[idx, :]
            keypoint_sample = matched_query_keypoints[idx, :]

            if use_p3p:
                # Backproject keypoints to unit bearing vectors
                
                normalized_bearings = np.matmul(K_inv, np.transpose(np.concatenate((keypoint_sample, np.ones((3,1))), axis=1)))
                for ii in range(3):
                    normalized_bearings[:,ii] /= np.linalg.norm(normalized_bearings[:,ii])

                poses = p3p.p3p(np.transpose(landmark_sample), normalized_bearings)

                # Decode p3p output
                R_C_W_guess = np.zeros((3,3,4))
                t_C_W_guess = np.zeros((3,1,4))
                for ii in range(4):
                    R_W_C_ii = np.real(poses[:, (1+ii*4):(4+ii*4)])
                    t_W_C_ii = np.real(poses[:, 4*ii])

                    R_C_W_guess[:,:,ii] = np.transpose(R_W_C_ii)
                    t_C_W_guess[:,:,ii] = -np.matmul(np.transpose(R_W_C_ii), t_W_C_ii[:,np.newaxis])

                M_C_W_guess[:,:3,:] = R_C_W_guess
                M_C_W_guess[:,-1,:] = t_C_W_guess[:,0,:]

            else:
                M_C_W_guess[:,:,0], _ = dlt.estimate_pose_dlt(keypoint_sample, landmark_sample) # first argument is the 2D correspondence point; second argument is the 3D correspondence point

            # Compute the differences between the matched query keypoints and the projected points
            # Matched query keypoints come from matching the keypoints (e.g. Harris corners) between subsequent frames (hypothesis)
            # Projected points come from projecting the 3D landmarks into the camera's image frame (model)
            projected_points = dlt.project_points(\
                np.matmul(\
                    np.concatenate(\
                        (corresponding_landmarks, np.ones((corresponding_landmarks.shape[0], 1))), axis=1), M_C_W_guess[:,:,0].transpose())) # p_landmarks,C = R_C_W * P_landmarks,W + t_C_W

            difference = matched_query_keypoints - projected_points
            errors = np.sum(np.square(difference), axis=1)
            is_inlier = errors < self.pixel_tolerance**2

            # If we use p3p, also consider inliers for the alternative solutions
            if use_p3p:
                for alt_idx in range(3):
                    projected_points = dlt.project_points(\
                        np.matmul(\
                            np.concatenate(\
                                (corresponding_landmarks, np.ones((corresponding_landmarks.shape[0], 1))), axis=1), M_C_W_guess[:,:,1+alt_idx].transpose()))
                    
                    difference = matched_query_keypoints - projected_points
                    errors = np.sum(np.square(difference), axis=1)
                    alternative_is_inlier = errors < self.pixel_tolerance**2

                    if np.count_nonzero(alternative_is_inlier) > np.count_nonzero(is_inlier):

                        is_inlier = alternative_is_inlier
            
            if tweaked_for_more:
                min_inlier_count = 30
            else:
                min_inlier_count = 6

            # Compute the number of inliers; update the inlier mask
            if np.count_nonzero(is_inlier) > max_num_inliers and np.count_nonzero(is_inlier) >= min_inlier_count:

                max_num_inliers = np.count_nonzero(is_inlier)
                
                best_inlier_mask = is_inlier

            if adaptive:
                # estimate of the outlier ratio
                outlier_ratio = 1 - max_num_inliers / is_inlier.shape[0]

                # formula to compute number of iterations from estimated outliers
                # ratio
                confidence = 0.95
                upper_bound_on_outlier_ratio = 0.90
                outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
                self.num_iterations = math.log10(1 - confidence) / math.log10(1 - (1 - outlier_ratio)**self.k)

                # cap the number of iterations at 15000
                self.num_iterations = min(15000, self.num_iterations)

            num_iteration_history.append(self.num_iterations)
            max_num_inliers_history.append(max_num_inliers)

            i += 1

        if max_num_inliers == 0:
            R_C_W = np.eye(3)
            t_C_W = np.zeros((3,1))

        else:
            M_C_W, _ = dlt.estimate_pose_dlt(matched_query_keypoints[best_inlier_mask > 0,:], corresponding_landmarks[best_inlier_mask > 0,:])
            R_C_W = M_C_W[:, :3]
            t_C_W = M_C_W[:, -1]

        if adaptive:
            print("     Adaptive RANSAC: Needed {} iterations to converge.".format(str(i-1)))
            print("     Adaptive RANSAC: Estimated outliers: {}%".format(str(int(100*outlier_ratio))))

        self.inlier_mask = best_inlier_mask

        return R_C_W, t_C_W, best_inlier_mask, np.array(max_num_inliers_history), np.array(num_iteration_history)

    def plot_matches(self, ax=plt.gca(), mask=False):

        if mask:
            corresponding_matches = self.all_matches[self.all_matches > 0]
            self.detector.plot_matches(corresponding_matches[self.inlier_mask > 0], self.matched_query_keypoints[self.inlier_mask > 0, :], self.database_keypoints, ax)
        else:
            self.detector.plot_matches(self.all_matches, self.query_keypoints, self.database_keypoints, ax)