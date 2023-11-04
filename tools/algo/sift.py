from tools.algo.harris import Harris
from matplotlib import image
import numpy as np
import math
import cv2

class SIFT():
    def __init__(self, params):

        self.params = params

        return

    def compute_image_pyramid(self, img):

        img_pyramid = [img]
        for i in range(1, self.params["num_octaves"]):
            
            # The input to each octave is an image downsampled by 2^o where o is the index of the octave
            dsize = (img_pyramid[i-1].shape[0] / 2, img_pyramid[i-1].shape[1] / 2) 

            img_pyramid.append(cv2.resize(img_pyramid[i-1], dsize))

        return img_pyramid

    def compute_blurred_images(self, img):
        
        # For each octave of the image pyramid apply Gaussian blurring recursively num_scales number of times and append the resulting blurred image to the pyramid
        volume = []
        for o in range(self.params["num_octaves"]):
            volume.append(np.zeros((img[o].shape[0], img[o].shape[1], self.params["num_scales"]+3)))

            for s in range(-1, self.params["num_scales"]+3):

                sigma = 2 ** (s / self.params["num_scales"]) * self.params["sigma"]

                volume[o][:,:,s+1] = cv2.GaussianBlur(img[o], self.params["kernel_size"], sigma, borderType=cv2.BORDER_DEFAULT)

        return volume

    def compute_DoGs(self, blurred_imgs):

        # Compute the difference of Gaussians over the blurred image pyramids
        DoGs = []
        for i in range(self.params["num_octaves"]):
            DoG_shape = blurred_imgs[i].shape

            # Since we're computing the difference between adjacent octaves of the blurred image pyramid, the resulting DoG pyramid will be num_octaves-1 deep
            DoG_shape[2] -= 1

            DoG = np.zeros(DoG_shape)

            num_DoGs_per_octave = DoG.shape[2]

            for j in range(num_DoGs_per_octave-1):
                DoG[:, :, j] = np.abs(blurred_imgs[i][:, :, j+1] - blurred_imgs[i][:, :, j])

            DoGs.append(DoG)

        return DoGs

    def extract_keypoints_from_DoGs(self, DoGs):

        # Find the keypoints in the difference of Gaussians. A keypoint is defined as a voxel in the DoG that is higher than its neighbors in scale and space.
        keypoint_locations = []
        for i in range(self.params["num_octaves"]):
            DoG = DoGs[i]

            kernel = np.ones((3, 3, 3), np.uint8)

            DoG_max = cv2.dilate(DoG, kernel, iterations=1)

            is_keypoint = (DoG == DoG_max) & (DoG >= self.params["contrast_threshold"])

            is_keypoint[:, :, 0]= False

            is_keypoint[:, :, -1] = False

            keypoint_locations.append(np.unravel_index(SIFT.find(is_keypoint, lambda x: x == True) , is_keypoint.shape))

        return keypoint_locations

    def compute_descriptors(self, blurred_imgs, keypoint_locations):

        descriptors = []
        final_keypoint_locations = []

        # Multiplication by 1.5 taken from Lowe's paper
        gaussWindow = cv2.getGaussianKernel((16, 16), 16 * 1.5)
        for i in range(self.params["num_octaves"]):
            blurred_img = blurred_imgs[i]
            
            keypoint_location = keypoint_locations[i]

            # Only analyze relevant images
            _, relevant_img_indices = np.unique(keypoint_location[:, 2], return_index=True)

            for img_idx in relevant_img_indices:

                img = blurred_img[:, :, img_idx]

                is_keypoint_in_image = keypoint_location[:, 2] == img_idx

                image_keypoint_locations = keypoint_location[is_keypoint_in_image, :]

                image_keypoint_locations = image_keypoint_locations[:, :2]

                num_keypoints = image_keypoint_locations.shape[0]

                image_descriptors = np.zeros(num_keypoints, 128)

                is_valid = np.zeros((num_keypoints, 1)).astype(np.bool)

                (rows_img, cols_img) = img.shape

                # compute the magnitudes and directions of the gradients for the current image
                G_mag, G_dir = SIFT.imgradient(img)

                for corner_idx in range(len(num_keypoints)):

                    row = image_keypoint_locations[corner_idx, 1]
                    col = image_keypoint_locations[corner_idx, 2]

                    if row > 8 and col > 8 and row < rows_img - 7 and col < cols_img - 7:
                        is_valid[corner_idx] = True

                        Gmag_loc = G_mag[row-8:row+7, col-8:col+7]

                        G_mag_loc_w = Gmag_loc * gaussWindow

                        Gdir_loc = G_dir[row-8:row+7, col-8:col+7]

                        Gmag_loc_derotated_w = G_mag_loc_w
                        Gdir_loc_derotated = Gdir_loc

                        if self.params["rotation_inv"] == 1:
                            # compute dominant direction through looking at the  most common orientation in the histogram, spaced at 10 degree increments
                            angle_edges = np.linspace(-180, 180, num=37)

                            orient_hist = SIFT.weightedhistc(Gdir_loc.reshape((-1,1)), G_mag_loc_w.reshape((-1,1)))

                            max_orient_idx = np.argmax(orient_hist)

                            G_dir_loc_principal = (angle_edges[max_orient_idx] + angle_edges[max_orient_idx + 1]) / 2

                            # derotate patch
                            patch_derotated = SIFT.derotate_patch(img, (row, col), 16, G_dir_loc_principal)

                            G_mag_loc_derotated, G_dir_loc_derotated = SIFT.imgradient(patch_derotated)

                            G_mag_loc_derotated_w = G_mag_loc_derotated * gaussWindow

                        N_tmp = 1
                        for ix in range(4):
                            for iy in range(4):
                                N_w = SIFT.weightedhistc(np.reshape(G_dir_loc_derotated[4*ix-3:4*ix, 4*iy-3:4*iy], (1, 16)), np.reshape(G_mag_loc_derotated_w[4*ix-3:4*ix, 4*iy], (1, 16)), np.linspace(-180, 180, num=9))

                                image_descriptors[corner_idx, N_tmp:N_tmp+7] = N_w[0, :8]
                                N_tmp += 8

                # Adapt keypoint location such that they correspond to the original image dimensions
                image_keypoint_locations = image_keypoint_locations * 2**(i - 1)

                # Only store valid keypoints
                descriptors.append(image_descriptors[is_valid, :])

                final_keypoint_locations.append(image_keypoint_locations[is_valid, :])

        # Normalize the descriptors such that they have unit norm
        descriptors = np.asarray(descriptors)
        row_sums = np.linalg.norm(descriptors, ord=2, axis=1)
        descriptors = descriptors / row_sums[:, np.newaxis]

        final_keypoint_locations = np.asarray(final_keypoint_locations)

        return (descriptors, final_keypoint_locations)

    @staticmethod
    def derotate_patch(img, loc, patch_size, ori):

        patch_radius = int(patch_size / 2)
        derotated_patch = np.zeros(patch_size, patch_size)

        padded_img = Harris.padarray(img, patch_size)

        # compute derotated patch
        for px in range(patch_size):
            for py in range(patch_size):
                x_origin = px - 1 - patch_radius
                y_origin = py - 1 - patch_radius

                # rotate patch by angle -ori
                x_rotated = math.cos(np.pi * ori / 180.) * x_origin - math.sin(np.pi * ori / 180) * y_origin

                y_rotated = math.sin(np.pi * ori / 180) * x_origin + math.cos(np.pi * ori / 180) * y_origin

                # move coordinates to patch
                x_patch_rotated = loc[1] + x_rotated

                y_patch_rotated = loc[0] - y_rotated

                # sample image (using nearest neighbor sampling)
                derotated_patch[py, px] = padded_img[math.ceil(y_patch_rotated + patch_radius), math.ceil(x_patch_rotated + patch_radius)]

        return derotated_patch


    @staticmethod
    def weightedhistc(vals, weights, edges):

        Nedge = edges.shape[0]
        h = np.zeros(edges.shape)

        for n in range(Nedge-1):
            ind = SIFT.find(vals, lambda x: x > edges[n] & x < edges[n+1])

            if not ind: # if indices are empty...
                h[n] = np.sum(weights[ind])

        ind = SIFT.find(vals, lambda x: x == edges[-1])
        if not ind:
            h[Nedge] = np.sum(weights[ind])

        return h

    @staticmethod
    def imgradient(img):

        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

        magnitude = np.sqrt(sobel_x**2.0 + sobel_y**2.0)
        angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi # deg

        return (magnitude, angle)

    @staticmethod
    def find(arr, func):

        return [i for (i, val) in enumerate(arr) if func(val)]