function image = getImage(image_path, rescale_factor)
    image = im2double(imresize(rgb2gray(imread(image_path)),...
        rescale_factor));
end

