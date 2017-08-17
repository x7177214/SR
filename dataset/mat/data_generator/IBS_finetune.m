clc
clear
close

save_path = '../train/x2/ibs_finetune/';

if ~exist(save_path, 'dir')
  mkdir(save_path);
end

name = 'IBS32_0003';
im_gt = imread([name, '.bmp']);
name = 'FHD_0003';
im_lr = imread([name, '.bmp']);

im_gt = rgb2gray(im_gt);
im_lr = rgb2gray(im_lr);

im_gt = im2double(im_gt);
im_lr = im2double(im_lr);

im_lr = imresize(im_lr, 2, 'near');

num_samples = 1000;

scale_factor = 2;
save_bicubic = 1;
count = 0;

patch_size = 17 * scale_factor;

for i = 1:num_samples
    x = randi(2490-1344-patch_size)+1344-1;
    y = randi(1440-724-patch_size)+724-1;
    while(x<=2160 && x>=1674-patch_size && y<=1320 && y>=836-patch_size || mod(x, 2) ~= 1 || mod(y, 2) ~= 1)
        x = randi(2490-1344-patch_size)+1344;
        y = randi(1440-724-patch_size)+724;
    end
    
    anchor = [y, x];
    
    patch_gt = im_gt(anchor(1):anchor(1)+patch_size-1, anchor(2):anchor(2)+patch_size-1);   
    patch_lr = im_lr(anchor(1):anchor(1)+patch_size-1, anchor(2):anchor(2)+patch_size-1);
    
    patch_lr = imresize(patch_lr, 0.5, 'near');
    
    for d = 0:3
        degree = d*90;
    % (original size)
        patch_name = sprintf('%s%d', save_path, count);        
        patch = imrotate(patch_gt, degree);
        save(patch_name, 'patch');
    % (downsample)
        patch = imrotate(patch_lr, degree);
        save(sprintf('%s_%d', patch_name, scale_factor), 'patch');    
    % (restored image)
        if save_bicubic == 1
            patch_name2 = sprintf('%s%d%s', save_path, count, strcat('_', num2str(scale_factor), 'b'));  
            patch = imresize(patch, scale_factor, 'bicubic');
            save(patch_name2, 'patch');
        end
        count = count+1;

    % (original size)
        patch_name = sprintf('%s%d', save_path, count);        
        patch = fliplr(imrotate(patch_gt, degree));
        save(patch_name, 'patch');
    % (downsample)
        patch = fliplr(imrotate(patch_lr, degree));
        save(sprintf('%s_%d', patch_name, scale_factor), 'patch');    
    % (restored image)
        if save_bicubic == 1
            patch_name2 = sprintf('%s%d%s', save_path, count, strcat('_', num2str(scale_factor), 'b'));  
            patch = imresize(patch, scale_factor, 'bicubic');
            save(patch_name2, 'patch');
        end
        count = count+1;
    end
end

