clear
clc
close

load('0')
load('0_4')
load('0_4b')
load('0_4cbcr')

img_cb = img_cbcr(:, :, 1);
img_cr = img_cbcr(:, :, 2);

img_cb = img_cb * 255.0;
img_cr = img_cr * 255.0;
im2 = imread('1_230_res_plus_bic.bmp');
im2 = rgb2gray(im2);
img_y = double(im2);

%% A

R = img_y                                + 1.402 * (img_cr - 128);
G = img_y + (-0.344136) * (img_cb - 128) + (-0.714136) * (img_cr - 128);
B = img_y + 1.772 * (img_cb - 128);

full_img(:, :, 1) = R;
full_img(:, :, 2) = G;
full_img(:, :, 3) = B;

%% B

full_img(:, :, 1) = img_y;
full_img(:, :, 2) = img_cb;
full_img(:, :, 3) = img_cr;

full_img = ycbcr2rgb(full_img);

%% C vvvvv

img_y = img_y-16;
img_cb = img_cb-128;
img_cr = img_cr-128;

full_img(:,:,1) = 0.004566210045662 * img_y + 0.006258928969944 * img_cr;
full_img(:,:,2) = .00456621 * img_y -0.001536323686045 * img_cb -0.003188110949656 * img_cr;
full_img(:,:,3) = .00456621 * img_y + 0.007910716233555 * img_cb;

full_img = uint8(full_img*255.0);

%%

imshow(full_img);

%%

A = [65.481 128.553 24.966; -37.797 -74.203 112.0; 112.0 -93.786 -18.214];

