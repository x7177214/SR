function local_var_map = var_map(img, kernel_size)

    
    padding_size = floor(kernel_size/2);
    padded_im = padarray(img, [padding_size, padding_size], 'symmetric'); % padding 0
    
    
    [gx, gy] = gradient(padded_im);
    var_map = abs(gx) + abs(gy);
    
    local_var_map = filter2([1, 1, 1; 1, 1, 1; 1, 1, 1], var_map); % local var map
    local_var_map = local_var_map(1+padding_size: end - padding_size, 1+padding_size: end - padding_size);
    
end