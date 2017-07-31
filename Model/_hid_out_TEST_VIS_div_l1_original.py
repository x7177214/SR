import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob
import os
import re
import scipy.io
import pickle
import argparse
import matplotlib.pyplot as plt
from PSNR import psnr255
from MODEL_div_l1_original_hid_out import model
from Subpixel import sb_test

### controller #####################  
SCALE_FACTOR = 2 # scale factor
DATASET = "small_d" # Dataset you want to infer
EPOCH = 40 # Model epoch you want to infer
CKPT_NAME = "x2_div_l1_original_0.25tv_ON_LAPSR_manga" # Model name
GPU_ID = "1" # required @ office
####################################  

SAVEIMAGE_OR_DISPLAYPSNR = 0
'''
0: Save Image, 1: Display
'''

DATA_PATH = "../dataset/mat/test/x%d/"%SCALE_FACTOR
CHECKPOINTS_PATH = "./checkpoints/" + CKPT_NAME
IMAGE_FORMAT = "*.bmp" #*.jpg or *.bmp
SAVE_PATH = "./result/" + CKPT_NAME

# set GPU 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_ID
from tensorflow.python.client import device_lib
print device_lib.list_local_devices()

def get_img_list(data_path):
    l = glob.glob(os.path.join(data_path, "*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    test_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + "_2.mat"):
                test_list.append([f,f[:-4] + "_2.mat",
                                    f[:-4] + "_2b.mat",
                                    f[:-4] + "_2cbcr.mat",
                                    2])
            if os.path.exists(f[:-4] + "_3.mat"):
                test_list.append([f,f[:-4] + "_3.mat",
                                    f[:-4] + "_3b.mat",
                                    f[:-4] + "_3cbcr.mat",
                                    3])
            if os.path.exists(f[:-4] + "_4.mat"):
                test_list.append([f,f[:-4] + "_4.mat",
                                    f[:-4] + "_4b.mat",
                                    f[:-4] + "_4cbcr.mat",
                                    4])
    return test_list


def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset + batch_size]
    input_list = []
    bic_list = []
    gt_list = []
    cbcr_list = []
    scale_list = []
    for pair in target_list:

        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        input_img = scipy.io.loadmat(pair[1])["img"]
        bic_img = scipy.io.loadmat(pair[2])["img_b"]
        cbcr_img = scipy.io.loadmat(pair[3])["img_cbcr"]

        gt_list.append(gt_img)
        input_list.append(input_img)
        bic_list.append(bic_img)
        cbcr_list.append(cbcr_img)
        scale_list.append(pair[4])
    return input_list, bic_list, gt_list, cbcr_list, scale_list

def Montage(v, ix, iy, ch, cy, cx, p = 0) :
    v = np.reshape(v, (iy, ix, ch))
    ix += 2
    iy += 2
    npad = ((1,1), (1,1), (0,0))
    v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
    v = np.reshape(v,(iy, ix, cy, cx)) 
    v = np.transpose(v, (2, 0, 3, 1)) #cy,iy,cx,ix
    v = np.reshape(v, (cy*iy, cx*ix))
    return v

def Save_images(imgs, layer_idx, cy, cx):
    # path = SAVE_PATH +'_'+ DATASET+'/'+layer_idx
    path = SAVE_PATH +'_'+ DATASET + '/hid'
    if not os.path.exists(path):
        os.makedirs(path)

    imgs = np.array(imgs, dtype=np.float32)
    imgs = imgs[0, 0, :, :, :]

    # normalized to [0.0 ~ 1.0]
    Mins = np.min(imgs, axis = 0)
    Mins = np.min(Mins, axis = 0)

    imgs -= Mins

    MAXs = np.max(imgs, axis = 0)
    MAXs = np.max(MAXs, axis = 0)

    imgs /= MAXs 

    # to [0.0 ~ 255.0]
    imgs *= 255.0
    imgs = np.clip(imgs, 0.0, 255.0)
    
    print imgs.shape

    IMG = Montage(imgs, imgs.shape[1], imgs.shape[0], imgs.shape[2], cy, cx)

    result = Image.fromarray(np.uint8(IMG), mode='L')
    result.save(path + '/%s.bmp' % layer_idx)

    # for i in range(imgs.shape[2]):
        # result = Image.fromarray(np.uint8(imgs[:, :, i]), mode='L')
        # result.save(path + '/%d_%d.bmp' % (i, imgs.shape[2]))
    return

def test_with_sess(epoch, ckpt_path, data_path, sess, shared_model):
    folder_list = glob.glob(os.path.join(data_path, DATASET))
    print 'folder_list', folder_list

    input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1)) 
    output_tensor, weights, loss_v_l1, f1, f2, f3, f4 = shared_model(input_tensor, 0, SCALE_FACTOR, False)

    saver = tf.train.Saver(weights)
    tf.global_variables_initializer().run()
    saver.restore(sess, ckpt_path)

    for folder_path in folder_list:
        print folder_path
        img_list = get_img_list(folder_path)
        for i in range(len(img_list)):
            print("TESTING IMAGE [%02d/%02d]"%(i+1,len(img_list)))
            
            input_list, bic_list, gt_list, cbcr_list, scale_list = get_test_image(img_list, i, 1)
            
            img_small = input_list[0]
            img_3b = bic_list[0]
            img_raw = gt_list[0]
            img_cbcr = cbcr_list[0]

            y = img_small.reshape((1, img_small.shape[0], img_small.shape[1], 1))

            # save feature maps
            fm1 = sess.run([f1], feed_dict={input_tensor: y})
            fm2 = sess.run([f2], feed_dict={input_tensor: y})
            fm3 = sess.run([f3], feed_dict={input_tensor: y})
            fm4 = sess.run([f4], feed_dict={input_tensor: y})
            Save_images(fm1, 'L1', 8, 4)
            Save_images(fm2, 'L2', 1, 5)
            Save_images(fm3, 'L3', 1, 5)
            Save_images(fm4, 'L4', 8, 4)
            y = sess.run([output_tensor], feed_dict={input_tensor: y})
            Save_images(y, 'L5', SCALE_FACTOR, SCALE_FACTOR)

            # Final result
            y = sb_test(y, SCALE_FACTOR)
            y_res = y

            img_3b = img_3b.reshape((y.shape[0], y.shape[1], 1))

            y = (y + img_3b) * 255.0 
            img_cbcr = img_cbcr * 255.0
                
            METHOD = 'm3'
            result = np.concatenate((y, img_cbcr), axis=2)
            tmp = np.copy(result)

            tmp[:, :, 0] = result[:, :, 0] # Y
            tmp[:, :, 1] = result[:, :, 1] - 128.0 # Cb
            tmp[:, :, 2] = result[:, :, 2] - 128.0 # Cr

            k1 = 1.0
            k2 = 0.0
            k3 = 1.402
            k4 = -0.344136
            k5 = -0.714136
            k6 = 1.772
            k7 = 0.0

            result[:, :, 0] = k1 * tmp[:, :, 0] + k2 * tmp[:, :, 1] + k3 * tmp[:, :, 2]
            result[:, :, 1] = k1 * tmp[:, :, 0] + k4 * tmp[:, :, 1] + k5 * tmp[:, :, 2]
            result[:, :, 2] = k1 * tmp[:, :, 0] + k6 * tmp[:, :, 1] + k7 * tmp[:, :, 2]
            
            result = np.clip(result, 0.0, 255.0)
            result = Image.fromarray(np.uint8(result), mode='RGB')
            
            # Residual map
            y_res = y_res[:, :, 0]
            y_res = y_res*255.0
            y_res = np.clip(y_res, 0.0, 255.0)
            result2 = Image.fromarray(np.uint8(y_res), mode='L')

            # Bicubic map
            img_3b = img_3b[:, :, 0]
            img_3b = img_3b*255.0
            img_3b = np.clip(img_3b, 0.0, 255.0)
            result3 = Image.fromarray(np.uint8(img_3b), mode='L')

            #Save the result image
            if SAVEIMAGE_OR_DISPLAYPSNR == 0:
                if not os.path.exists(SAVE_PATH +'_'+ DATASET):
                    os.makedirs(SAVE_PATH +'_'+ DATASET)
                result.save(SAVE_PATH +'_'+ DATASET + '/hid' + '/id%d_epoch%d_%s_%s.bmp' % (i, epoch, METHOD, CKPT_NAME))
                result2.save(SAVE_PATH +'_'+ DATASET + '/hid' + '/id%d_RES_epoch%d_%s_%s.bmp' % (i, epoch, METHOD, CKPT_NAME))
                result3.save(SAVE_PATH +'_'+ DATASET + '/hid' + '/id%d_BIC_epoch%d_%s_%s.bmp' % (i, epoch, METHOD, CKPT_NAME))
            
            #PSNR and PLOT
            else:
                fig = plt.figure(figsize=(15, 5))
                a = fig.add_subplot(1, 3, 1)
                a.set_title('GT')
                img_raw = img_raw * 255
                im = np.concatenate(( img_raw.reshape((img_raw.shape[0], img_raw.shape[1], 1)), img_cbcr), axis=2)
                im = Image.fromarray(np.uint8(result), mode='RGB')
                plt.imshow(im, interpolation="none")
                a = fig.add_subplot(1, 3, 2)
                img_3b = img_3b * 255
                a.set_title('Interpolation %03f' % psnr255(img_raw, img_3b.reshape(img_3b.shape[0], img_3b.shape[1])))
                img_3b[np.where(img_3b < 0)] = 0
                img_3b[np.where(img_3b > 255)] = 255
                im_s = np.concatenate(( img_3b.reshape((img_3b.shape[0], img_3b.shape[1], 1)), img_cbcr), axis=2)
                im_s = Image.fromarray(np.uint8(im_s), mode='YCbCr')
                plt.imshow(im_s, interpolation="none")
                a = fig.add_subplot(1, 3, 3)
                a.set_title('RESULT %03f' % psnr255(img_raw, y.reshape(y.shape[0], y.shape[1])))
                plt.imshow(result, interpolation="none")
                plt.show()

if __name__ == '__main__':
    model_list = sorted(glob.glob(CHECKPOINTS_PATH + "/epoch_*"))
    model_list = [fn for fn in model_list if not os.path.basename(
        fn).endswith("meta")]
    model_list = [fn for fn in model_list if not os.path.basename(
        fn).endswith("index")]
    model_list = [fn.split('.data')[0] for fn in model_list]
  
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    with tf.Session(config=config) as sess:
        
        shared_model = tf.make_template('shared_model', model)

        for model_ckpt in model_list:
            epoch = int(model_ckpt.split('epoch_')[-1].split('.ckpt')[0])
            if epoch == EPOCH:
                print "Testing model:", model_ckpt
                test_with_sess(epoch, model_ckpt, DATA_PATH, sess, shared_model)
