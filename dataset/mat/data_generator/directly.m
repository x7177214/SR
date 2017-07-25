% Set5 Set14 Urban100 B100
target = 'nova_0.5_partial'; %name of testing dataset
dataDir = fullfile('../../raw/', target);

count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];
folder = fullfile('../test/x2', [target '_d']); %save path
mkdir(folder);
scale_factor = 2;

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
    
    img_raw = im2double(img_raw);
    
    if size(img_raw,3)==3
    %method 1
%         img_raw = rgb2ycbcr(img_raw);
%         img_cbcr = img_raw(:,:,2:3);
%         img_raw = img_raw(:,:,1);

    %method 2
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
            
    end
    
    img_size = size(img_raw);
    
%     img_2 = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
%     img_3 = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
%     img_4 = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    
%     img_raw = img_raw - 0.5;
%     img_raw = img_raw .* 0.5;
%     img_raw = img_raw +0.5;
    

    img = img_raw;
    
    img_b = imresize(img, scale_factor,'bicubic');
    
    img_cbcr = imresize(img_cbcr, scale_factor,'bicubic');
    
    patch_name = sprintf('%s/%d',folder,count);
    
    save(sprintf('%s', patch_name), 'img_raw')
    
    save(sprintf('%s_%d', patch_name, scale_factor), 'img');
    
    save(sprintf('%s_%db', patch_name, scale_factor), 'img_b');
    
    save(sprintf('%s_%dcbcr', patch_name, scale_factor), 'img_cbcr');
    
    clear img_cbcr;
    
    count = count + 1;
    
    display(sprintf('%d/ %d',f_iter, numel(f_lst)));
end
