clear all;
close all;

% Randomly chosen parameters that seem to work well - can you find better
% ones?
corner_patch_size = 9;
harris_kappa = 0.08;
num_keypoints = 200;
nonmaximum_supression_radius = 8;
descriptor_radius = 9;
match_lambda = 4;

img = imread('../data/000000.png');

%% Part 1 - Calculate Corner Response Functions
% Shi-Tomasi
shi_tomasi_scores = shi_tomasi(img, corner_patch_size);
assert(min(size(shi_tomasi_scores) == size(shi_tomasi_scores)));

% Harris
harris_scores = harris(img, corner_patch_size, harris_kappa);
assert(min(size(harris_scores) == size(harris_scores)));

figure('Color', 'w');
subplot(2, 2, 1);
imshow(img);
subplot(2, 2, 2);
imshow(img);

subplot(2, 2, 3);
imagesc(shi_tomasi_scores);
title('Shi-Tomasi Scores');
daspect([1 1 1]);

axis off;

subplot(2, 2, 4);
imagesc(harris_scores);
title('Harris Scores');
daspect([1 1 1]);

axis off;


%% Part 2 - Select keypoints

keypoints = selectKeypoints(...
    harris_scores, num_keypoints, nonmaximum_supression_radius);
figure(2);
imshow(img);
hold on;
plot(keypoints(2, :), keypoints(1, :), 'rx', 'Linewidth', 2);

%% Part 3 - Describe keypoints and show 16 strongest keypoint descriptors

descriptors = describeKeypoints(img, keypoints, descriptor_radius);
figure(3);
for i = 1:16
    subplot(4, 4, i);
    patch_size = 2 * descriptor_radius + 1;
    imagesc(uint8(reshape(descriptors(:,i), [patch_size patch_size])));
    axis equal;
    axis off;
end

%% Part 4 - Match descriptors between first two images
img_2 = imread('../data/000001.png');
harris_scores_2 = harris(img_2, corner_patch_size, harris_kappa);
keypoints_2 = selectKeypoints(...
    harris_scores_2, num_keypoints, nonmaximum_supression_radius);
descriptors_2 = describeKeypoints(img_2, keypoints_2, descriptor_radius);

matches = matchDescriptors(descriptors_2, descriptors, match_lambda);

figure(4);
imshow(img_2);
hold on;
plot(keypoints_2(2, :), keypoints_2(1, :), 'rx', 'Linewidth', 2);
plotMatches(matches, keypoints_2, keypoints);

%% Part 5 - Match descriptors between all images
figure(5);
img_indices = 0:199;
clear prev_desc
for i = img_indices
    img = imread(sprintf('../data/%06d.png',i));
    imshow(img); hold on;
    
    scores = harris(img, corner_patch_size, harris_kappa);
    kp = selectKeypoints(...
        scores, num_keypoints, nonmaximum_supression_radius);
    plot(kp(2, :), kp(1, :), 'rx', 'Linewidth', 2);
    
    desc = describeKeypoints(img, kp, descriptor_radius);
    if (exist('prev_desc', 'var'))
        matches = matchDescriptors(desc, prev_desc, match_lambda);
        plotMatches(matches, kp, prev_kp);
    end
    
    prev_kp = kp;
    prev_desc = desc;
    hold off;
    pause(0.01);
end