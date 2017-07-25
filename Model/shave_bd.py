import numpy as np

def shave_bd(img, bd):
    '''
    Description:
        crop image boundaries for 'bd' pixels
    Input:
        - img   : input image
        - bd    : pixels to be cropped

    Output:
        - img   : output image
    '''

    img = img[0+bd:0-bd, 0+bd:0-bd]
    return img