clc
clear
close

img_raw = imread('white.jpg');

img_raw = rgb2ycbcr(img_raw);
img_raw = im2double(img_raw);

Y = img_raw(:, :, 1);
Cb = img_raw(:, :, 2);
Cr = img_raw(:, :, 3);

save('dataY2', 'Y');
save('dataCb2', 'Cb');
save('dataCr2', 'Cr');

%% YCbCr formula by hand
clc
clear
close

img_raw = imread('white.jpg');

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

R = double(img_raw(:, :, 1));
G = double(img_raw(:, :, 2));
B = double(img_raw(:, :, 3));

Y   = 0     +  k1 * R + k2 * G + k3 * B;
Cb = 128 +  k4 * R + k5 * G + k6 * B;
Cr  = 128 +  k7 * R + k8 * G + k9 * B;

% im_new = zeros(size(img_raw, 1), size(img_raw, 2), size(img_raw, 3));

Y = Y/255.0;
Cb = Cb/255.0;
Cr = Cr/255.0;

gg(:, :, 1) = Cb;
gg(:, :, 2) = Y;

save('dataY', 'Y');
save('dataCb', 'Cb');
save('dataCr', 'Cr');

%%

clear
clc
close


