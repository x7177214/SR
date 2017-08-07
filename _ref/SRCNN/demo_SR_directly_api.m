% =========================================================================
% Test code for Super-Resolution Convolutional Neural Networks (SRCNN)
%
% Reference
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, 
%   in Proceedings of European Conference on Computer Vision (ECCV), 2014
%
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks,
%   arXiv:1500.00092
%
% Chao Dong
% IE Department, The Chinese University of Hong Kong
% For any question, send email to ndc.forward@gmail.com
% =========================================================================

close all;
clear all;
clc

%% set parameters
% up_scale = 3;
% model = 'model\9-5-5(ImageNet)\x3.mat';
% up_scale = 2;
% model = 'model\9-1-5(ImageNet)\x2.mat';
% up_scale = 2;
% model = './model/9-1-5(91 images)/x2.mat';
up_scale = 2;
model = 'model\9-5-5(ImageNet)\x2.mat'; 
% up_scale = 4;
% model = 'model\9-5-5(ImageNet)\x4.mat';

%% read images

target = 'nova_sub4'; %name of testing dataset
dataDir = fullfile('./', target);

f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];
folder = fullfile(['./x', num2str(up_scale), '_',target, '_d']); %save path

folder1 = [folder, '/SRCNN_955/'];

mkdir(folder);
mkdir(folder1);

%% do SR on images one by one

for f_iter = 1:numel(f_lst)

    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir, f_info.name);
    disp(f_path);
    
    im = imread(f_path);

%% work on illuminance only

    if size(im,3)>1
        %im = RGBtoYCBCR(im);
        im = rgb2ycbcr(im);
        cb = im(:, :, 2);
        cr = im(:, :, 3);
        im = im(:, :, 1);
    end

    %% bicubic interpolation
    im = single(im)/255.0;
    cb = double(cb);
    cr = double(cr);
    
    im_b = imresize(im, up_scale, 'bicubic');
    cb_b = imresize(cb, up_scale, 'bicubic');
    cr_b = imresize(cr, up_scale, 'bicubic');

    %% SRCNN

    im_h = SRCNN(model, im_b);

    %% merge Y Cb and Cr

    im_h = uint8(im_h * 255.0);

    im_H = zeros(size(im_h, 1), size(im_h, 2), 3);

    im_H(:, :, 1) = im_h;
    im_H(:, :, 2) = uint8(cb_b);
    im_H(:, :, 3) = uint8(cr_b);

    im_H = uint8(im_H);
    
    %% YCbCr 2 RGB

    %im_h = YCBCRtoRGB(im_H);
    im_h2 = ycbcr2rgb(im_H);

    %% show results

    %figure, imshow(im_h); title('SRCNN Reconstruction');
    
    %split_name = strsplit(f_info.name, '.');
    imwrite(im_h2, [folder1, f_info.name, '_SRCNN_x2_955' '.bmp']);
    
    clear im_h;
    clear im_H;
    clear im_b;
    clear cb_b;
    clear cr_b;
    clear cb;
    clear cr;
    
    display(sprintf('%d/ %d',f_iter, numel(f_lst)));
end
