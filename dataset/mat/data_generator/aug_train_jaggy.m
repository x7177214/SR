% USAGE : 
% Transform the images to YCbCr channels, and extract Y channel only.
% Then clip the images to many patches, and save the 
% {rotated 0, 90 ,180, 270} +
% {horizontal filpping : yes, no} version 
% of these image patch pairs (
%                 groundtruth : XXX.mat
%                 LR : XXX_scale_factor.mat
%                 restored image : XXX_b.mat
%                 )

% restored image :
%     gt image -> bicubic small image -> bicubic original size image

% The LR is produced by downsampling the groundtruth using the method of bicubic
   
clc
clear
close all
dataDir = '../../raw/LAPSR_21manga109';
save_path = '../train/x2/LAPSR_manga_jaggy/'; % a '/' must be added at end of the path

stride_factor = 9;

scale_factor = 2;
save_bicubic = 1;
over_scale = 1.2; % over down scale factor

count = 0;

f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];

patch_size = 17 * scale_factor;
stride = stride_factor * scale_factor;

if ~exist(save_path, 'dir')
  mkdir(save_path);
end

% RGB 2 YCbCr
k1 = 0.299;
k2 = 0.587;
k3 = 0.114;

for f_iter = 1:numel(f_lst)
    

    
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    img_raw = imread(f_path);
    
    %%%%%% rgb 2 ycbcr %%%%%%
    
    % method 1 
        %img_raw = rgb2ycbcr(img_raw);
        %img_raw = im2double(img_raw(:,:,1));
    
    % method 2
        R = double(img_raw(:, :, 1));
        G = double(img_raw(:, :, 2));
        B = double(img_raw(:, :, 3));

        Y   = 0 + k1 * R + k2 * G + k3 * B;
        img_raw = Y / 255.0;
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    
    img_size = size(img_raw);
    x_size = (img_size(2)-patch_size)/stride+1;
    y_size = (img_size(1)-patch_size)/stride+1;
    
    for x = 0:x_size-1
        for y = 0:y_size-1
         
            x_coord = x*stride; y_coord = y*stride; 
            
            
            for d = 0:3
                
                degree = d * 90;
            
    % (original size)
                patch_name = sprintf('%s%d', save_path, count);        
                patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), degree);
                save(patch_name, 'patch');

    % (downsample)
                patch = imresize(patch,1/scale_factor/over_scale,'bicubic');
                patch = imresize(patch,[17, 17],'near');

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
                patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), degree));
                save(patch_name, 'patch');

    % (downsample)
                patch = imresize(patch,1/scale_factor/over_scale,'bicubic');
                patch = imresize(patch,[17, 17],'near');

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
    end
    
    display(sprintf('%d/ %d',f_iter, numel(f_lst)));    
    
end
