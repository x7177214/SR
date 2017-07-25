% USAGE:
% 
%     input: (in jpg, png, bmp, ...)
%         image
% 
%     output: (in .mat)
%         GT: (X.mat)   
%             original image in Y channel in value [0, 1]
%         LR: (X_SCALE.mat)
%             GT image downsampled by 'bicubic'
%         HR: (X_SCALEb.mat)
%             LR image upsampled by 'bicubic'
%         CbCr: (X_SCALEcbcr.mat)
%             HR image in the version of CbCr channels other than Y channel
%             
% !!! mod crop and shave_bd are used !!!
% 

% --------------------------
% CONTROLLER
%
% Set5 Set14 Urban100 B100
target = 'Set14'; %name of testing dataset
scale_factor = 4;
%
%--------------------------

dataDir = fullfile('../../raw', target);

count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];
folder = fullfile(['../test/x', num2str(scale_factor)], target); %save_path

mkdir(folder);

% RGB 2 YCbCr
k1 = 0.299;
k2 = 0.587;
k3 = 0.114;

k4 = -0.168736;
k5 = -0.331264;
k6 = 0.5;

k7 = 0.5;
k8 = -0.418688;
k9 = -0.081312;

for f_iter = 1:numel(f_lst)

    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    disp(f_path);
    img_raw = imread(f_path);
    

    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    img_raw = im2double(img_raw); %[0 ~ 1]
    
    % crop the image to the sizes of multiples of scale factor
%     method 1
    img_raw = modcrop(img_raw, scale_factor);
%     method 2
    img_raw = img_raw(1:height-mod(height,scale_factor),1:width-mod(width,scale_factor),:);
   
    if size(img_raw,3)==3
          
        % method 1
%             img_raw = rgb2ycbcr(img_raw);
%             img_cbcr = img_raw(:,:,2:3);
%             img_raw = img_raw(:,:,1);

        % method 2
                img_raw = img_raw * 255.0; %[0 ~ 255]
                
                R = double(img_raw(:, :, 1));
                G = double(img_raw(:, :, 2));
                B = double(img_raw(:, :, 3));
                
                Y   = 0 + k1 * R + k2 * G + k3 * B;%[0 ~ 255]
                Cb = 128 +  k4 * R + k5 * G + k6 * B;%[0 ~ 255]
                Cr  = 128 +  k7 * R + k8 * G + k9 * B;%[0 ~ 255]
                
                img_raw = Y / 255.0; %[0 ~ 1]
                
                img_cbcr(:, :, 1) = Cb / 255.0; %[0 ~ 1]
                img_cbcr(:, :, 2) = Cr / 255.0; %[0 ~ 1]
        
    else % gray image
                img_raw = img_raw * 255.0; %[0 ~ 255]
                
                R = double(img_raw(:, :, 1));
                G = double(img_raw(:, :, 1));
                B = double(img_raw(:, :, 1));
                
                Y   = 0 + k1 * R + k2 * G + k3 * B;%[0 ~ 255]
                Cb = 128 +  k4 * R + k5 * G + k6 * B;%[0 ~ 255]
                Cr  = 128 +  k7 * R + k8 * G + k9 * B;%[0 ~ 255]
                
                img_raw = Y / 255.0; %[0 ~ 1]
                
                img_cbcr(:, :, 1) = Cb / 255.0; %[0 ~ 1]
                img_cbcr(:, :, 2) = Cr / 255.0; %[0 ~ 1]
        
%         img_raw = rgb2ycbcr(repmat(img_raw, [1 1 3]));
    end
    
    img_size = size(img_raw);
    
%     img_2 = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
%     img_3 = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
%     img_4 = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    
    img = imresize(img_raw,1/scale_factor,'bicubic');
    
%     img_b = imresize(img,[img_size(1),img_size(2)],'bicubic');
    img_b = imresize(img, scale_factor, 'bicubic');
    
    
    
    patch_name = sprintf('%s/%d',folder,count);
    
    save(patch_name, 'img_raw');
%     save(sprintf('%s_2', patch_name), 'img_2');
    save(sprintf('%s_%d', patch_name, scale_factor), 'img');
%     save(sprintf('%s_4', patch_name), 'img_4');
    
    save(sprintf('%s_%db', patch_name, scale_factor), 'img_b');
    
    img_cbcr = imresize(img_cbcr,1/scale_factor,'bicubic');
    img_cbcr = imresize(img_cbcr, scale_factor,'bicubic');
    save(sprintf('%s_%dcbcr', patch_name, scale_factor), 'img_cbcr');
    
    clear img_cbcr;
    
    count = count + 1;
    
    display(sprintf('%d/ %d',f_iter, numel(f_lst)));
end
