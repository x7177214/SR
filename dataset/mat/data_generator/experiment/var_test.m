clc
clear
close


kernel_size = 3;


padding_size = floor(kernel_size/2);

im_b = imread('9.bmp');
% im_b = imresize(im_b, 2, 'bicubic');
im_sr = imread('x2_9.bmp');

im_b = rgb2gray(im_b);
im_sr = rgb2gray(im_sr);

im_b = im2double(im_b);
im_sr = im2double(im_sr);

var_map_b = var_map(im_b, kernel_size);
var_map_sr = var_map(im_sr, kernel_size);

var_map_b = uint8(var_map_b * 255.0);
var_map_sr = uint8(var_map_sr * 255.0);

imwrite(var_map_b, '9var_map_b.bmp');
imwrite(var_map_sr, '9var_map_sr.bmp');
imwrite(abs(var_map_sr - var_map_b), '9var_map_diff.bmp');
