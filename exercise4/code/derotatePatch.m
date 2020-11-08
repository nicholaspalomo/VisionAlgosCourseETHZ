function derotated_patch = derotatePatch(img, loc, patch_size, ori)

patch_radius = patch_size / 2;
derotated_patch = zeros(patch_size,patch_size);

padded_img = padarray(img, [patch_radius, patch_radius]);
% compute derotated patch  
for px=1:patch_size
    for py=1:patch_size
      x_origin = px - 1 - patch_radius;
      y_origin = py - 1 - patch_radius;

      % rotate patch by angle -ori
      x_rotated = cos(pi*ori/180) * x_origin - sin(pi*ori/180) * y_origin;
      y_rotated = sin(pi*ori/180) * x_origin + cos(pi*ori/180) * y_origin;

      % move coordinates to patch
      x_patch_rotated = loc(2) + x_rotated;
      y_patch_rotated = loc(1) - y_rotated;

      % sample image (using nearest neighbor sampling as opposed to more
      % accuracte bilinear sampling)
      derotated_patch(py, px) = padded_img(ceil(y_patch_rotated+patch_radius), ...
          ceil(x_patch_rotated+patch_radius));
    end
end
      
end

