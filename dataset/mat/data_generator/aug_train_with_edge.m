clear
close all
dataDir = '/home/peteryu/SR/DataSet/91G100';


% gt : xxx.mat 
% gt edge : xxx_edge.mat
% input X :xxx_f.mat  ,f = scale


count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
save_path = 'train/91G100_x4_2s/';

scale_factor = 4;
weight = 0.8;

patch_size = 17 * scale_factor;
stride = 9 * scale_factor;

if ~exist(save_path, 'dir')
  mkdir(save_path);
end

for f_iter = 1:numel(f_lst)
    
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    img_raw = imread(f_path);
    img_raw = rgb2ycbcr(img_raw);
    img_raw = im2double(img_raw(:,:,1));
    
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
            
            %%% gt image %%%
            patch_name = sprintf('%s%d', save_path, count);        
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(patch_name, 'patch'); % figure(); imshow(patch);
            
            %%% gt edge image %%%
            patch_name2 = sprintf('%s%d%s', save_path, count, '_edge');  
            edge_patch = canny_edge(patch, weight);
            save(patch_name2, 'edge_patch');
            
            %%% input X %%%
            patch = imresize(patch,1/scale_factor,'bicubic');
            save(sprintf('%s_%d', patch_name, scale_factor), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('%s%d', save_path, count);
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
            save(patch_name, 'patch');
            
            patch_name2 = sprintf('%s%d%s', save_path, count, '_edge');  
            edge_patch = canny_edge(patch, weight);
            save(patch_name2, 'edge_patch');
            
            patch = imresize(patch,1/scale_factor,'bicubic');
            save(sprintf('%s_%d', patch_name, scale_factor), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('%s%d', save_path, count);
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
            save(patch_name, 'patch');
            
            patch_name2 = sprintf('%s%d%s', save_path, count, '_edge');  
            edge_patch = canny_edge(patch, weight);
            save(patch_name2, 'edge_patch');
            
            patch = imresize(patch,1/scale_factor,'bicubic');
            save(sprintf('%s_%d', patch_name, scale_factor), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('%s%d', save_path, count);
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
            save(patch_name, 'patch');
            
            patch_name2 = sprintf('%s%d%s', save_path, count, '_edge');  
            edge_patch = canny_edge(patch, weight);
            save(patch_name2, 'edge_patch');
            
            patch = imresize(patch,1/scale_factor,'bicubic');
            save(sprintf('%s_%d', patch_name, scale_factor), 'patch');
            
            count = count+1;
 
            patch_name = sprintf('%s%d', save_path, count);
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180);
            save(patch_name, 'patch');
            
            patch_name2 = sprintf('%s%d%s', save_path, count, '_edge');  
            edge_patch = canny_edge(patch, weight);
            save(patch_name2, 'edge_patch');
            
            patch = imresize(patch,1/scale_factor,'bicubic');
            save(sprintf('%s_%d', patch_name, scale_factor), 'patch');
            
            count = count+1;
 
            patch_name = sprintf('%s%d', save_path, count);
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(patch_name, 'patch');
            
            patch_name2 = sprintf('%s%d%s', save_path, count, '_edge');  
            edge_patch = canny_edge(patch, weight);
            save(patch_name2, 'edge_patch');
            
            patch = imresize(patch,1/scale_factor,'bicubic');
            save(sprintf('%s_%d', patch_name, scale_factor), 'patch');
            
            count = count+1;
      
            patch_name = sprintf('%s%d', save_path, count);
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270);
            save(patch_name, 'patch');
            
            patch_name2 = sprintf('%s%d%s', save_path, count, '_edge');  
            edge_patch = canny_edge(patch, weight);
            save(patch_name2, 'edge_patch');
            
            patch = imresize(patch,1/scale_factor,'bicubic');
            save(sprintf('%s_%d', patch_name, scale_factor), 'patch');
            
            count = count+1;
        
            patch_name = sprintf('%s%d', save_path, count);
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270));
            save(patch_name, 'patch');
            
            patch_name2 = sprintf('%s%d%s', save_path, count, '_edge');  
            edge_patch = canny_edge(patch, weight);
            save(patch_name2, 'edge_patch');
            
            patch = imresize(patch,1/scale_factor,'bicubic');
            save(sprintf('%s_%d', patch_name, scale_factor), 'patch');
            
            count = count+1;
        end
    end
    
    display(sprintf('%d/ %d',f_iter, numel(f_lst)));    
    
end