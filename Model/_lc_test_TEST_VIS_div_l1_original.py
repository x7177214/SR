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
from MODEL_div_l1_original import model
from Subpixel import sb_test

SCALE_FACTOR = 2
DATASET = "nova_d"
EPOCH = 200 # Epoch you want to test
SAVEIMAGE_OR_DISPLAYPSNR = 0
'''
0: Save Image, 1: Display
'''

DATA_PATH = "../dataset/mat/test/x%d/"%SCALE_FACTOR
CHECKPOINTS_PATH = "./checkpoints/x%d_div_l1_original_plus_manga"%SCALE_FACTOR
IMAGE_FORMAT = "*.bmp" #*.jpg or *.bmp
SAVE_PATH = "./result/x%d_div_l1_original_plus_manga"%SCALE_FACTOR

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

def div_norm(input_, k_size=3):
    mean = tf.nn.avg_pool(input_, ksize=[1, k_size, k_size, 1],
                            strides=[1, 1, 1, 1], padding='SAME')
    mean = tf.reduce_mean(mean, axis=3, keep_dims=True)
    v = input_ - mean
    var = tf.nn.avg_pool(v ** 2, ksize=[1, k_size, k_size, 1],
                            strides=[1, 1, 1, 1], padding="SAME")
    var = tf.reduce_mean(var, axis=3, keep_dims=True)
    sigma = tf.constant(0.5)
    return v / tf.sqrt(var + sigma**2)

def local_var(input_, k_size=3):
    mean = tf.nn.avg_pool(input_, ksize=[1, k_size, k_size, 1],
                            strides=[1, 1, 1, 1], padding='SAME')
    mean = tf.reduce_mean(mean, axis=3, keep_dims=True)
    v = input_ - mean
    var = tf.nn.avg_pool(v ** 2, ksize=[1, k_size, k_size, 1],
                            strides=[1, 1, 1, 1], padding="SAME")
    return tf.reduce_mean(var, axis=3, keep_dims=True)

def test_with_sess(epoch, ckpt_path, data_path, sess, shared_model):
    folder_list = glob.glob(os.path.join(data_path, DATASET))
    print 'folder_list', folder_list

    input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1)) 
    output_tensor, weights, loss_v_l1 = shared_model(input_tensor, 0, SCALE_FACTOR, False)

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

            y = sess.run([output_tensor], feed_dict={input_tensor: y})
            y = sb_test(y, SCALE_FACTOR)

            img_3b = img_3b.reshape((y.shape[0], y.shape[1], 1))

            y = y + img_3b

            y = y - 0.5
            y = y * 2
            y = y + 0.5

            y = y * 255.0 
            img_cbcr = img_cbcr * 255.0

            # method_1
                # y[np.where(y < 0.0)] = 0.0
                # y[np.where(y > 255.0)] = 255.0 
                # img_cbcr[np.where(img_cbcr < 0.0)] = 0.0
                # img_cbcr[np.where(img_cbcr > 255.0)] = 255.0   
                
                # result = np.concatenate((y, img_cbcr), axis=2)
                # result = Image.fromarray(np.uint8(result), mode='YCbCr')
                # result = result.convert('RGB')

            # method_2
                # tmp = np.copy(result)

                # tmp[:, :, 0] = result[:, :, 0] - 16.0 # Y
                # tmp[:, :, 1] = result[:, :, 1] - 128.0 # Cb
                # tmp[:, :, 2] = result[:, :, 2] - 128.0 # Cr

                # k1 = 0.004566210045662
                # k2 = 0.006258928969944
                # k3 = -0.001536323686045
                # k4 = -0.003188110949656
                # k5 = 0.007910716233555

                # result[:, :, 0] = k1 * tmp[:, :, 0] + k2 * tmp[:, :, 2]
                # result[:, :, 1] = k1 * tmp[:, :, 0] + k3 * tmp[:, :, 1] + k4 * tmp[:, :, 2]
                # result[:, :, 2] = k1 * tmp[:, :, 0] + k5 * tmp[:, :, 1]
                
                # result[:, :, 0] = result[:, :, 0] * 255.0
                # result[:, :, 1] = result[:, :, 1] * 255.0 
                # result[:, :, 2] = result[:, :, 2] * 255.0

                # result[np.where(result < 0.0)] = 0.0
                # result[np.where(result > 255.0)] = 255.0 
                # result = Image.fromarray(np.uint8(result), mode='RGB')

            # method_2(api)
            # result = np.concatenate((y, img_cbcr), axis=2)
            # result = ycbcr2rgb(result)
            # result = Image.fromarray(np.uint8(result), mode='RGB')
            
            # method_3: the same formula used in method 1, but implemented by hand
              
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
            
            result[np.where(result < 0.0)] = 0.0
            result[np.where(result > 255.0)] = 255.0

            result = Image.fromarray(np.uint8(result), mode='RGB')
            
            #Save the result image
            if SAVEIMAGE_OR_DISPLAYPSNR == 0:
                if not os.path.exists(SAVE_PATH +'_'+ DATASET):
                    os.makedirs(SAVE_PATH +'_'+ DATASET)
                result.save(SAVE_PATH +'_'+ DATASET + '/%d_%d_%s.bmp' % (i, epoch, METHOD))
            
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
