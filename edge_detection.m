function edge_map = edge_detection(im1)

    % USAGE : find strong edges in a image, used by "aug_train_with_edge_ver2.m"
    % Input : gray image
    % Output : edge map

    %im1 = rgb2gray(im1);
    %im1 = im2double(im1);

    im1 = padarray(im1, [1, 1], 'symmetric'); % padding 0

    x1 = filter2([-1, 1], im1); % right gradient
    x2 = filter2([1, -1], im1); % left gradient

    y1 = filter2([-1; 1], im1); % down gradient
    y2 = filter2([1; -1], im1); % up gradient

    % collect the 4 gradients
    G = zeros(size(im1, 1), size(im1, 2), 4);
    G(:, :, 1) = x1; G(:, :, 2) = x2;
    G(:, :, 3) = y1; G(:, :, 4) = y2;

    G = G(2:end-1, 2:end-1, :); % de-padding
    
    % find the max abs gradient beyond the 4 gradients
    G = abs(G);
    G = max(G, [], 3);

    T = 0.4; % theshold of gradient value 
    W = 7; % width of dilation structure ele. 

    mask = G > T;
    se = strel('square',W); 
    mask2 = imdilate(mask, se);
    mask2 = mask2 + 0.0; % 2 double
    edge_map = mask2;

end
