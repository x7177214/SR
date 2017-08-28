clc
clear
close

im = imread('e.jpg');
im = rgb2gray(im);
im = im2double(im);

h1 = fspecial('sobel');
im_1 = imfilter(im,h1,'replicate'); 

h2 = [2 1 0; 1 0 -1;0 -1 -2];
im_2 = imfilter(im,h2,'replicate'); 


H = fspecial('average', 3);
im_11a = imfilter(im_1,H,'replicate');
im_22a = imfilter(im_2,H,'replicate'); 

mean = im_11a + im_22a;
mean = mean * 0.5;

v_1 =  im_1 - mean;
v_2 =  im_2 - mean;

v_1sq = v_1.^2.0;
v_2sq = v_2.^2.0;

v_1sq_a = imfilter(v_1sq,H,'replicate');
v_2sq_a = imfilter(v_2sq,H,'replicate'); 

var = v_1sq_a + v_2sq_a;
var = var * 0.5;

sig = 0.7;

new_im_1 = v_1 ./ sqrt(var + sig) ;
new_im_2 = v_2 ./ sqrt(var + sig) ;

subplot(2, 3, 1)
imshow(im);
title('original')

subplot(2, 3, 2 )
imshow(im_1);
title('0 degree Sobel response')

subplot(2, 3, 3)
imshow(im_2);
title('45 degree Sobel response')

subplot(2, 3, 5 )
imshow(new_im_1);
title('decorrelated 0 degree Sobel response')

subplot(2, 3, 6)
imshow(new_im_2);
title('decorrelated 45 degree Sobel response')

