function out = YCBCRtoRGB(im)
    % USAGE : YCbCr to RGB

    % YCbCr 2 RGB
    
    Y = im(:, :, 1);
    Cb = im(:, :, 2);
    Cr = im(:, :, 3);
    
    Cb = Cb - 128.0;
    Cr = Cr - 128.0;

    k1 = 1.0;
    k2 = 0.0;
    k3 = 1.402;
    k4 = -0.344136;
    k5 = -0.714136;
    k6 = 1.772;
    k7 = 0.0;

    R = k1 * Y + k2 * Cb + k3 * Cr;
    G = k1 * Y + k4 * Cb + k5 * Cr;
    B = k1 * Y + k6 * Cb + k7 * Cr;

    out = im;
    out(:, :, 1) = R;
    out(:, :, 2) = G;
    out(:, :, 3) = B;

    out(out < 0.0) = 0.0;
    out(out > 255.0) = 255.0;

    out = uint8(out);
end