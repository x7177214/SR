
import numpy as np
import math

def psnr(target, ref, scale):
    #assume RGB image
    target_data = np.array(target, dtype='double')
    #target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref, dtype='double')
    #ref_data = ref_data[scale:-scale, scale:-scale]

    correction = 0.8588 # analog Y to digital Y
    diff = (ref_data - target_data) * correction


    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)


def psnr255(img1, img2):
    
    img1 = np.array(img1, dtype='double')
    img2 = np.array(img2, dtype='double')

    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
