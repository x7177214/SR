function edge_img = canny_edge(img, weight)

% weight : the weight for calculation MSE in non-edge region.
%                 The weight in the edge region is 1.

    edge_img = edge(img, 'Canny');

    edge_img = double(edge_img);

    edge_img(edge_img == 0) = weight;

end
