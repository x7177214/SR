import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob
import os
import re
from PSNR import psnr
import scipy.io
import pickle
from MODEL_div_l1 import model
import argparse
from Subpixel import sb_test
from shave_bd import shave_bd

SCALE_FACTOR = 4
DATASET = "Set14"#"testForValidation" # all "*"

#DATA_PATH = "./data/test/x%d"%SCALE_FACTOR
DATA_PATH = "../dataset/mat/test/x%d"%SCALE_FACTOR

CHECKPOINTS_PATH = "./checkpoints/x%d_div_l1_gamma_beta_plus_more_manga"%SCALE_FACTOR
PSNR_PATH = "psnr/x%d_div_l1_gamma_beta_plus_more_manga"%SCALE_FACTOR

if not os.path.exists(PSNR_PATH):
    os.makedirs(PSNR_PATH)

def get_img_list(data_path):
    l = glob.glob(os.path.join(data_path, "*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    test_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + "_2.mat"):
                test_list.append([f, f[:-4] + "_2.mat", f[:-4] + "_2b.mat", 2])
            if os.path.exists(f[:-4] + "_3.mat"):
                test_list.append([f, f[:-4] + "_3.mat", f[:-4] + "_3b.mat", 3])
            if os.path.exists(f[:-4] + "_4.mat"):
                test_list.append([f, f[:-4] + "_4.mat", f[:-4] + "_4b.mat", 4])
    return test_list


def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset + batch_size]
    input_list = []
    bic_list = []
    gt_list = []
    scale_list = []
    for pair in target_list:

        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        input_img = scipy.io.loadmat(pair[1])["img"]
        bic_img = scipy.io.loadmat(pair[2])["img_b"]

        gt_list.append(gt_img)
        input_list.append(input_img)
        bic_list.append(bic_img)
        scale_list.append(pair[3])
    return input_list, bic_list, gt_list, scale_list


def test_with_sess(epoch, ckpt_path, data_path, sess, shared_model):
    folder_list = glob.glob(os.path.join(data_path, DATASET))
    print 'folder_list', folder_list
    input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
    
    output_tensor, weights, loss_v_l1 = shared_model(input_tensor, 0, SCALE_FACTOR, False)
    
    saver = tf.train.Saver(weights)
    tf.global_variables_initializer().run()
    saver.restore(sess, ckpt_path)

    psnr_dict = {}
    for folder_path in folder_list:
        psnr_list = []
        img_list = get_img_list(folder_path)
        psnr_avg_bic = 0.0
        psnr_avg_sr = 0.0
        for i in range(len(img_list)):
            input_list, bic_list, gt_list, scale_list = get_test_image(img_list, i, 1)
            input_y = input_list[0]
            bic_y = bic_list[0]
            gt_y = gt_list[0]

            sr_y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(
                input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
            sr_y = sb_test(sr_y, SCALE_FACTOR)
            sr_y = np.resize(
                sr_y, (sr_y.shape[0], sr_y.shape[1]))
            sr_y = bic_y + sr_y

            # Remove border pixels as some methods (e.g., A+) do not predict border pixels
            gt_y = shave_bd(gt_y, SCALE_FACTOR)
            bic_y = shave_bd(bic_y, SCALE_FACTOR)
            sr_y = shave_bd(sr_y, SCALE_FACTOR)

            psnr_bic = psnr(bic_y, gt_y, scale_list[0])
            psnr_sr = psnr(sr_y, gt_y, scale_list[0])
            psnr_avg_bic = psnr_avg_bic + psnr_bic
            psnr_avg_sr = psnr_avg_sr + psnr_sr
            #print "[%d/%d] bic: %.2f\tSR: %.2f" % (i+1,len(img_list),psnr_bic, psnr_sr)
            psnr_list.append([psnr_bic, psnr_sr, scale_list[0]])
        print("AVG:  bic: %.2f\tSR: %.2f"%(psnr_avg_bic/len(img_list),psnr_avg_sr/len(img_list)))
        psnr_dict[os.path.basename(folder_path)] = psnr_list
    with open(PSNR_PATH + '/%s' % os.path.basename(ckpt_path), 'wb') as f:
        pickle.dump(psnr_dict, f)

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
            print "Testing model:", model_ckpt
            test_with_sess(80, model_ckpt, DATA_PATH, sess, shared_model)
