function out = RGBtoYCBCR(im)
    % USAGE : RGB to YCbCr

    k1 = 0.299;
    k2 = 0.587;
    k3 = 0.114;

    k4 = -0.168736;
    k5 = -0.331264;
    k6 = 0.5;

    k7 = 0.5;
    k8 = -0.418688;
    k9 = -0.081312;

    R = double(im(:, :, 1));
    G = double(im(:, :, 2));
    B = double(im(:, :, 3));

    Y   = 0 + k1 * R + k2 * G + k3 * B;%[0 ~ 255]
    Cb = 128 +  k4 * R + k5 * G + k6 * B;%[0.5 ~ 255.5]
    Cr  = 128 +  k7 * R + k8 * G + k9 * B;%[0.5 ~ 255.5]

    out = im;
    out(:, :, 1) = Y;
    out(:, :, 2) = Cb;
    out(:, :, 3) = Cr;
end
