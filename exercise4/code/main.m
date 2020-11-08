clear all
close all
clc

rotation_inv = 1;
rotation_img2_deg = 0;

num_scales = 3; % Scales per octave.
num_octaves = 5; % Number of octaves.
sigma = 1.6;
contrast_threshold = 0.04;
image_file_1 = 'images/img_1.jpg';
image_file_2 = 'images/img_2.jpg';
rescale_factor = 0.2; % Rescaling of the original image for speed.

left_img = getImage(image_file_1, rescale_factor);
right_img = getImage(image_file_2, rescale_factor);

% to test rotation invariance of SIFT

if rotation_img2_deg ~= 0
    right_img = imrotate(right_img, rotation_img2_deg);
end

images = {left_img, right_img};

kpt_locations = cell(1, 2);
descriptors = cell(1, 2);

for img_idx = 1:2
    % Write code to compute:
    % 1)    image pyramid. Number of images in the pyarmid equals
    %       'num_octaves'.
    % 2)    blurred images for each octave. Each octave contains
    %       'num_scales + 3' blurred images.
    % 3)    'num_scales + 2' difference of Gaussians for each octave.
    % 4)    Compute the keypoints with non-maximum suppression and
    %       discard candidates with the contrast threshold.
    % 5)    Given the blurred images and keypoints, compute the
    %       descriptors. Discard keypoints/descriptors that are too close
    %       to the boundary of the image. Hence, you will most likely
    %       lose some keypoints that you have computed earlier.
end

% Finally, match the descriptors using the function 'matchFeatures' and
% visualize the matches with the function 'showMatchedFeatures'.
% If you want, you can also implement the matching procedure yourself using
% 'knnsearch'.