from __future__ import print_function

import time
import os
import sys
import logging
import json
import random
import tensorflow as tf
import numpy as np
import cv2
import data.data_loader as loader
from models.cgan_model import cgan
from models.ops import *

imh = 700
imw = 900
IN_CHANEL = 1

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
gpu_options = tf.GPUOptions(allow_growth=True)


def linear_decay(initial=0.0001, step=0, start_step=150, end_step=300):
    '''
    return decayed learning rate
    It becomes 0 at end_step
    '''
    decay_period = end_step - start_step
    step_decay = (initial - 0.0) / decay_period
    update_step = max(0, step - start_step)
    current_value = max(0, initial - (update_step) * step_decay)
    return current_value


def train(args):
    # assume there is a batch data pair:

    data_path = args.data_path_train
    pair_path = loader.read_data_path_custom(data_path)
    # print(pair_path)
    dataset = loader.batch_generator(pair_path, batch_size=args.batch_num)
    num_dataset = len(pair_path)
    print("the training sample num: ", num_dataset)
    num_batch = int(np.ceil(num_dataset / args.batch_num))
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    model = cgan(sess, args)
    model.build_model()
    model.sess.run(tf.global_variables_initializer())
    model.load_weights(args.checkpoint_dir)

    for iter in range(args.epoch):
        learning_rate = linear_decay(1e-2, iter)
        # for i, data in enumerate(dataset):
        for i in range(num_batch):
            batch_pair_paths = next(dataset)
            blur_imgs, sharp_imgs = loader.quick_read_frames_pair(batch_pair_paths, args.img_h, args.img_w)
            # blur_img, real_img = loader.read_image_pair(data,
            #                                             resize_or_crop=args.resize_or_crop,
            #                                             image_size=(args.img_h, args.img_w))

            feed_dict = {model.input['blur_img']: blur_imgs, \
                         model.input['real_img']: sharp_imgs, \
                         model.learning_rate: learning_rate}

            loss_G, adv_loss, ssim_loss, mae_loss = model.run_optim_G(feed_dict=feed_dict)
            logging.info('%d epoch,  %d batch, Generator Loss:  %f, add loss: %f, ssim_loss: %f, mae_loss: %f',
                         iter, i, loss_G, adv_loss, ssim_loss, mae_loss)

            # Ready for Training Discriminator
            # loss_D, loss_disc, loss_gp = -1e6, -1e6, -1e6
            for _ in range(args.iter_disc):
                loss_D, loss_disc, loss_gp = model.run_optim_D(feed_dict=feed_dict, with_image=args.tf_image_monitor)

            logging.info('%d epoch,  %d  batch, Discriminator  Loss:  %f, loss_disc:  %f, gp_loss: %f', iter, i, loss_D,
                         loss_disc, loss_gp)

        # if (iter + 1) % 1 == 0 or iter == (args.epoch - 1):
        model.save_weights(args.checkpoint_dir, model.global_step)

    # logging.info("[!] test started")
    # dataset_test = loader.read_data_path_custom(args.data_path_test)
    #
    # for i, data in enumerate(dataset):
    #     if not os.path.exists('./test_result'):
    #         os.mkdir('./test_result')
    #     blur_img, real_img = loader.quick_read_frames_pair(dataset_test)
    #     feed_dict_G = {model.input['blur_img']: blur_img}
    #     G_out = model.G_output(feed_dict=feed_dict_G)
    #     cv2.imwrite('./test_result/' + str(i) + '_blur.png', blur_img)
    #     cv2.imwrite('./test_result/' + str(i) + '_real.png', real_img)
    #     cv2.imwrite('./test_result/' + str(i) + '_gen.png', G_out[0])
    #     logging.info("Deblur Image is saved (%d/%d) ", i, len(dataset))
    # logging.info("[*] test done")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--iter_gen', type=int, default=1)
    parser.add_argument('--iter_disc', type=int, default=3)
    parser.add_argument('--batch_num', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--data_path_train', type=str,
                        default='../../conv_gru_result_rain/gru_tf_data/3_layer_7_5_3/Test/Deblur/Train/')
    parser.add_argument('--data_path_test', type=str,
                        default='../../conv_gru_result_rain/gru_tf_data/3_layer_7_5_3/Test/Deblur/Test/')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--model_name', type=str, default='RAINGAN.model')
    parser.add_argument('--summary_dir', type=str, default='./summaries/')
    parser.add_argument('--data_name', type=str, default='RAIN')
    parser.add_argument('--tf_image_monitor', type=bool, default=True)

    parser.add_argument('--resize_or_crop', type=str, default='resize')
    parser.add_argument('--img_h', type=int, default=imh)
    parser.add_argument('--img_w', type=int, default=imw)
    parser.add_argument('--img_c', type=int, default=IN_CHANEL)

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    logging.getLogger("DeblurGAN_TRAIN.*").setLevel(level)

    train(args)
