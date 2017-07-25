im_b = imread('4.bmp');
im_b = imresize(im_b, 4, 'bicubic');
im_sr = imread('4sr.bmp');

im_b = rgb2gray(im_b);
im_sr = rgb2gray(im_sr);

subplot(2, 1, 1)
histogram(im_b)
subplot(2, 1, 2)
histogram(im_sr)