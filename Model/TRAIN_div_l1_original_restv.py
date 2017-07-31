import os
import glob
import re
import signal
import sys
import argparse
import threading
from random import shuffle
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
from MODEL_div_l1_original_res_xav import model # original : no gamma and beta ; with l1 regularization
#######Controler########
SCALE_FACTOR = 2
LAMBDA = 1.75 # total_loss = loss + LAMBDA * loss_tv + L1_regularization
TRAIN_DATA = 'LAPSR_manga'
########################
CHECKPOINT_PATH = "./checkpoints/x%d_div_l1_original_%.2frestv_ON_%s" % (SCALE_FACTOR, LAMBDA, TRAIN_DATA)
TRAIN_DATA_PATH = "../dataset/mat/train/x%d/%s" % (SCALE_FACTOR, TRAIN_DATA)

IN_IMG_SIZE = (17, 17)
OUT_IMG_SIZE = (17 * SCALE_FACTOR, 17 * SCALE_FACTOR)
BATCH_SIZE = 128
MOMENTUM = 0.9
USE_ADAM_OPT = True
if USE_ADAM_OPT:
    BASE_LR = 2e-4 # origin : 1e-4
else:
    BASE_LR = 0.1
LR_RATE = 0.02
LR_STEP_SIZE = 300
START_EPOCH = 0
MAX_EPOCH = 301
USE_QUEUE_LOADING = True
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

file = open(CHECKPOINT_PATH + '/Train_detail.txt', 'w')
file.write('CHECKPOINT_PATH: %s' % CHECKPOINT_PATH + '\n')
file.write('TRAIN_DATA_PATH: %s' % TRAIN_DATA_PATH + '\n')
file.write('SCALE_FACTOR: %d' % SCALE_FACTOR + '\n')
file.write('BATCH_SIZE: %d' % BATCH_SIZE + '\n')
file.write('USE_ADAM_OPT %s' % USE_ADAM_OPT + '\n')
file.write('MOMENTUM: %f' % MOMENTUM + '\n')
file.write('BASE_LR: %f' % BASE_LR + '\n')
file.write('LR_RATE: %f' % LR_RATE + '\n')
file.write('LR_STEP_SIZE: %d' % LR_STEP_SIZE + '\n')
file.write('MAX_EPOCH: %d' % MAX_EPOCH + '\n')
file.close()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

def get_train_list(data_path):
    l = glob.glob(os.path.join(data_path, "*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    print 'Get training list', len(l)
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + "_2.mat"):
                train_list.append([f, f[:-4] + "_2.mat", f[:-4] + "_2b.mat"])
                # train_list.append([f, f[:-4] + "_2.mat"])
            if os.path.exists(f[:-4] + "_3.mat"):
                train_list.append([f, f[:-4] + "_3.mat", f[:-4] + "_3b.mat"])
                # train_list.append([f, f[:-4] + "_3.mat"])
            if os.path.exists(f[:-4] + "_4.mat"):
                train_list.append([f, f[:-4] + "_4.mat", f[:-4] + "_4b.mat"])
                # train_list.append([f, f[:-4] + "_4.mat"])
    return train_list

def get_image_batch(train_list, offset, batch_size):
    target_list = train_list[offset:offset + batch_size]
    input_list = []
    gt_list = []
    cbcr_list = []
    for pair in target_list:
        input_img = scipy.io.loadmat(pair[1])['patch']
        gt_img = scipy.io.loadmat(pair[0])['patch']
        input_list.append(input_img)
        gt_list.append(gt_img)
    input_list = np.array(input_list)
    input_list.resize([BATCH_SIZE, IN_IMG_SIZE[1], IN_IMG_SIZE[0], 1])
    gt_list = np.array(gt_list)
    gt_list.resize([BATCH_SIZE, OUT_IMG_SIZE[1], OUT_IMG_SIZE[0], 1])
    return input_list, gt_list

if __name__ == '__main__':
    train_list = get_train_list(TRAIN_DATA_PATH)

    if not USE_QUEUE_LOADING:
        print "not use queue loading, just sequential loading..."
        train_input = tf.placeholder(tf.float32, shape=(
            BATCH_SIZE, IN_IMG_SIZE[0], IN_IMG_SIZE[1], 1))
        train_gt = tf.placeholder(tf.float32, shape=(
            BATCH_SIZE, OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], 1))
    else:
        print "use queue loading"
        train_bic_single = tf.placeholder(
            tf.float32, shape=(OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], 1))
        train_input_single = tf.placeholder(
            tf.float32, shape=(IN_IMG_SIZE[0], IN_IMG_SIZE[1], 1))
        train_gt_single = tf.placeholder(
            tf.float32, shape=(OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], 1))
        q = tf.FIFOQueue(1000, [tf.float32, tf.float32, tf.float32], [
                         [OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], 1],
                         [IN_IMG_SIZE[0], IN_IMG_SIZE[1], 1],
                         [OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], 1]])
        enqueue_op = q.enqueue([train_bic_single, train_input_single, train_gt_single])
        train_bic, train_input, train_gt = q.dequeue_many(BATCH_SIZE)

    shared_model = tf.make_template('shared_model', model)
    train_output, weights, loss_v_l1, res = shared_model(train_input, train_bic, SCALE_FACTOR, True)

    # [LOSS] L2 norm
    # loss = tf.reduce_sum(tf.nn.l2_loss(train_output - train_gt))
    
    # [LOSS] Chabonnier
    loss = tf.reduce_sum(tf.sqrt(tf.square(train_output - train_gt)+tf.square(1e-3)))
        
    # [LOSS] image total variation
    loss_tv = tf.reduce_sum(tf.image.total_variation(res))

    # [LOSS] DN feature map L1 
    loss_v_l1 = tf.reduce_mean(loss_v_l1)

    loss = loss + LAMBDA * loss_tv + loss_v_l1


    #for w in weights:
    #    loss += tf.nn.l2_loss(w)*1e-4

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        BASE_LR, global_step * BATCH_SIZE, len(train_list) * LR_STEP_SIZE, LR_RATE, staircase=False)

    if USE_ADAM_OPT:
        optimizer = tf.train.AdamOptimizer(learning_rate)
        opt = optimizer.minimize(loss, global_step=global_step)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
        tvars = tf.trainable_variables()
        gvs = zip(tf.gradients(loss, tvars), tvars)
        norm = 0.1 / (learning_rate/BASE_LR)
        capped_gvs = [(tf.clip_by_norm(grad, norm), var) for grad, var in gvs]
        opt = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    saver = tf.train.Saver(weights, max_to_keep=0)

    shuffle(train_list)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        tf.global_variables_initializer().run()

        if model_path:
            print "restore model..."
            saver.restore(sess, model_path)
            print "Done"

        if USE_QUEUE_LOADING:
            def load_and_enqueue(coord, file_list, idx=0, num_thread=1):
                count = 0
                length = len(file_list)
                while not coord.should_stop():
                    i = (count * num_thread + idx) % length
                    bic_img = scipy.io.loadmat(file_list[i][2])['patch'].reshape(
                        [OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], 1])
                    input_img = scipy.io.loadmat(file_list[i][1])['patch'].reshape(
                        [IN_IMG_SIZE[0], IN_IMG_SIZE[1], 1])
                    gt_img = scipy.io.loadmat(file_list[i][0])['patch'].reshape(
                        [OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], 1])
                    sess.run(enqueue_op, feed_dict={train_bic_single: bic_img,
                            train_input_single: input_img, train_gt_single: gt_img})
                    count += 1

            coord = tf.train.Coordinator()
            num_thread = 1
            for i in range(num_thread):
                t = threading.Thread(target=load_and_enqueue, args=(
                    coord, train_list, i, num_thread))
                t.start()

        def signal_handler(signum, frame):
            print "stop training, save checkpoint..."
            saver.save(sess, CHECKPOINT_PATH + "/epoch_%03d.ckpt" %
                       epoch, global_step=global_step)
            coord.join()
            print "Done"
            sys.exit(1)
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)

        if USE_QUEUE_LOADING:
            for epoch in xrange(START_EPOCH, MAX_EPOCH):
                for step in range(len(train_list) // BATCH_SIZE):

                    l1, _, l, output, lr, g_step = sess.run(
                        [loss_v_l1, opt, loss, train_output, learning_rate, global_step])
                    print "[epoch %2.4f] loss %.4f l1 %.8f lr %.8f " % (epoch + (float(step) * BATCH_SIZE / len(train_list)), np.sum(l) / BATCH_SIZE, l1, lr)
                    
                if epoch % 5 == 0:
                    saver.save(sess, CHECKPOINT_PATH + "/epoch_%03d.ckpt" %
                               epoch, global_step=global_step)
        else:
            for epoch in xrange(START_EPOCH, MAX_EPOCH):
                for step in range(len(train_list) // BATCH_SIZE):
                    offset = step * BATCH_SIZE
                    input_data, gt_data = get_image_batch(
                        train_list, offset, BATCH_SIZE)
                    feed_dict = {train_input: input_data, train_gt: gt_data}
                    _, l, output, lr, g_step = sess.run(
                        [opt, loss, train_output, learning_rate, global_step], feed_dict=feed_dict)
                    print "[epoch %2.4f] loss %.4f\t lr %.5f" % (epoch + (float(step) * BATCH_SIZE / len(train_list)), np.sum(l) / BATCH_SIZE, lr)
                    del input_data, gt_data
                if epoch % 10 == 0:
                    saver.save(sess, CHECKPOINT_PATH + "/epoch_%03d.ckpt" %
                               epoch, global_step=global_step)

        coord.join()
        print "Reach Max Epoch, Done"
