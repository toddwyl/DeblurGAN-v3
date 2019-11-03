from __future__ import print_function

import os
import numpy as np
import cv2
import glob
import logging
import random
import numpy
import os
from concurrent.futures import ThreadPoolExecutor, wait
from imageio import imread
import tensorflow as tf

# np.set_printoptions(threshold=1e6)
imh = 700
imw = 900
IN_CHANEL = 1

color = {
    0: [0, 0, 0, 0],
    1: [0, 236, 236, 255],
    2: [1, 160, 246, 255],
    3: [1, 0, 246, 255],
    4: [0, 239, 0, 255],
    5: [0, 200, 0, 255],
    6: [0, 144, 0, 255],
    7: [255, 255, 0, 255],
    8: [231, 192, 0, 255],
    9: [255, 144, 2, 255],
    10: [255, 0, 0, 255],
    11: [166, 0, 0, 255],
    12: [101, 0, 0, 255],
    13: [255, 0, 255, 255],
    14: [153, 85, 201, 255],
    15: [255, 255, 255, 255],
    16: [0, 0, 0, 0]
}
# keys = list(color.keys())
# values = [color[k] for k in keys]
# table = tf.contrib.lookup.HashTable(
#   tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64, value_dtype=tf.float32), [0, 0, 0, 0]
# )

#
# H = 720
# W = 912


def read_data_path_custom(data_path, image_type='png'):
    pair_path = dir_image_pair(data_path)
    return pair_path


_imread_executor_pool = ThreadPoolExecutor(max_workers=8)
_imread_executor_pool_2 = ThreadPoolExecutor(max_workers=8)


def read_img(path, read_storage):
    img = numpy.asarray(imread(path), dtype=np.float32)
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    if img.shape[0] == 900:
        # read_storage[2:-2,14:-14,:] = img[100:-100,:,:]
        index_h = int((900 - 700) / 2)
        img = img[index_h:-index_h, :, :]
    # Normalization!!
    # img = img / 255.0 * 2.0 - 1.0
    img = map_class(img)  # map to 0-15
    img = img / 15. * 2. - 1.
    read_storage[:] = img[:]


def quick_read_frames(path_list, im_h, im_w):
    """Multi-thread Frame Loader

    Parameters
    ----------
    path_list : list
    im_h : height of image
    im_w : width of image

    Returns
    -------

    """
    img_num = len(path_list)
    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            print(path_list[i])
            raise IOError
    read_storage = numpy.zeros((img_num, im_h, im_w, IN_CHANEL), dtype=numpy.uint8)
    if img_num == 1:
        read_img(path_list[0], read_storage[0])
    else:
        future_objs = []
        for i in range(img_num):
            obj = _imread_executor_pool.submit(read_img, path_list[i], read_storage[i])
            future_objs.append(obj)
        wait(future_objs)
    return read_storage[...]


def quick_read_frames_pair(path_list_pair, im_h=imh, im_w=imw):
    """Multi-thread Frame Loader

    Parameters
    ----------
    path_list : list
    im_h : height of image
    im_w : width of image

    Returns
    -------

    """
    img_num = len(path_list_pair)
    for i in range(img_num):
        if not os.path.exists(path_list_pair[i][0]):
            print(path_list_pair[i][0])
            raise IOError
        if not os.path.exists(path_list_pair[i][1]):
            print(path_list_pair[i][1])
            raise IOError
    read_storage_blur = np.zeros((img_num, im_h, im_w, IN_CHANEL), dtype=np.float32)
    read_storage_sharp = np.zeros((img_num, im_h, im_w, IN_CHANEL), dtype=np.float32)
    if img_num == 1:
        read_img(path_list_pair[0][0], read_storage_blur[0])
        read_img(path_list_pair[0][1], read_storage_sharp[0])
    else:
        future_objs = []
        future_objs2 = []
        for i in range(img_num):
            obj = _imread_executor_pool.submit(read_img, path_list_pair[i][0], read_storage_blur[i])
            future_objs.append(obj)
        wait(future_objs)
        for i in range(img_num):
            obj2 = _imread_executor_pool_2.submit(read_img, path_list_pair[i][1], read_storage_sharp[i])
            future_objs2.append(obj2)
        wait(future_objs2)
    return read_storage_blur[...], read_storage_sharp[...]


# def read_data_path(data_path, name='GOPRO', image_type='png'):
#     dir_list = [dir for dir in glob.glob(data_path + '/*') if os.path.isdir(dir)]
#     image_pair_path = []
#     for i, dir in enumerate(dir_list):
#         if not name in dir:
#             dir_list.remove(dir)
#     dir_image_pair(dir_list[0])
#
#     for i, dir in enumerate(dir_list):
#         image_pair_path.extend(dir_image_pair(dir))
#     return image_pair_path


def dir_image_pair(dir_path, image_type='png'):
    blur_path = os.path.join(dir_path, 'blur')
    real_path = os.path.join(dir_path, 'sharp')
    # print("blur_path:", blur_path)
    blur_image_pathes = glob.glob(blur_path + '/*.' + image_type)
    real_image_pathes = glob.glob(real_path + '/*.' + image_type)
    assert len(blur_image_pathes) == len(real_image_pathes)
    pair_path = zip(blur_image_pathes, real_image_pathes)
    iter_pair_path = pair_path  # for iteration

    result = list(pair_path)

    for blur, real in iter_pair_path:
        name1 = blur.split('/')[-1]
        name2 = real.split('/')[-1]
        if name1 != name2:
            result.remove((blur, real))
            print("blur: %s, real: %s pair was removed in training data" % (name1, name2))
    return result


# def read_image_pair(pair_path, resize_or_crop=None, image_size=(imh, imw)):
#     # image_blur = cv2.imread(pair_path[0], cv2.IMREAD_COLOR)
#     # image_blur = image_blur / 255.0 * 2.0 - 1.0
#     # image_real = cv2.imread(pair_path[1], cv2.IMREAD_COLOR)
#     # image_real = image_real / 255.0 * 2.0 - 1.0
#
#     image_blur = cv2.imread(pair_path[0], cv2.IMREAD_GRAYSCALE)
#     print(image_blur.shape)
#     image_real = cv2.imread(pair_path[1], cv2.IMREAD_GRAYSCALE)
#     print(image_real.shape)
#     # cv2.imshow('image_blur2', image_blur)
#     if resize_or_crop != None:
#         assert image_size != None
#
#     if resize_or_crop == 'resize':
#         image_blur = cv2.resize(image_blur, image_size, interpolation=cv2.INTER_AREA)
#         image_real = cv2.resize(image_real, image_size, interpolation=cv2.INTER_AREA)
#     elif resize_or_crop == 'crop':
#         image_blur = cv2.crop(image_blur, image_size)
#         image_real = cv2.crop(image_real, image_size)
#     # else:
#     #     raise
#
#     if np.size(np.shape(image_blur)) == 3:
#         image_blur = np.expand_dims(image_blur, axis=0)
#     if np.size(np.shape(image_real)) == 3:
#         image_real = np.expand_dims(image_real, axis=0)
#
#     image_blur = np.array(image_blur, dtype=np.float32)
#     image_real = np.array(image_real, dtype=np.float32)
#     # cv2.imshow('image_blur2', image_blur)
#     return image_blur, image_real


# def read_image(path, resize_or_crop=None, image_size=(700, 900)):
#     # image = cv2.imread(path, cv2.IMREAD_COLOR)
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     # image = image/255.0 * 2.0 - 1.0
#
#     assert resize_or_crop != None
#     assert image_size != None
#
#     if resize_or_crop == 'resize':
#         image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
#     elif resize_or_crop == 'crop':
#         image = cv2.crop(image, image_size)
#
#     if np.size(np.shape(image)) == 3:
#         image = np.expand_dims(image, axis=0)
#
#     image = np.array(image, dtype=np.float32)
#     return image

def batch_generator(data, batch_size=8, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    random.seed(1000)
    if shuffle:
        random.shuffle(data)

    batch_count = -1
    num_data = len(data)
    while True:
        batch_count += 1
        start = batch_count * batch_size
        end = start + batch_size
        if end >= num_data:
            batch_count = -1
            end = num_data
        yield data[start:end]


def map_class(img):
    img = np.where(img >= 75, 15, img // 5)
    return img


def map_rule(num):
    if num >= 75:
        return 15
    else:
        return num // 5


def make_color(img):
    """Map each gray level pixel in origin image to RGBA space
    Parameter
    ---------
    img : ndarray (a gray level image)

    Returns
    ---------
    img : An Image object with RGBA mode

    """
    # color_map = form_color_map()
    h, w, _ = img.shape
    # img = tf.squeeze(img)
    # img = tf.convert_to_tensor(img)
    new_img = np.zeros((h, w, 4), dtype=np.int8)
    # new_img = tf.map_fn(lambda i: (color[i][0], color[i][1], color[i][2], color[i][3]), img)
    # new_img = tf.map_fn(lambda i: table.lookup(i), img)
    # print(new_img)
    for i in range(h):
        for j in range(w):
            # print(img[i, j, 0])
            new_img[i, j] = color[int(img[i, j, 0])]
    # print("done")
    # img = Image.fromarray(new_img, mode="RGBA")
    return new_img


if __name__ == '__main__':
    data_path = r'/media/todd/632DF6B97AFFCBE0/code/conv_gru_result_rain/gru_tf_data/3_layer_7_5_3/Test/Deblur_slim/Train'
    pair_path = read_data_path_custom(data_path)
    # print(pair_path)
    dataset = batch_generator(pair_path, batch_size=8)
    batch_pair_paths = next(dataset)
    # print(batch_pair_paths)
    print("paths:", batch_pair_paths)
    blur_imgs, sharp_imgs = quick_read_frames_pair(batch_pair_paths, imh, imw)
    print(blur_imgs.shape)
    # random.seed(1000)
    # random.shuffle(pair_path)
    # image1, image2 = read_image_pair(pair_path[0])
    # print(image1.shape)
    # # print("image_blur", image1)

    # blur_imgs = (blur_imgs + 1.) / 2. * 255.
    # sharp_imgs = (sharp_imgs + 1.) / 2. * 255.
    blur_imgs = map_class(img=blur_imgs)
    sharp_imgs = map_class(img=sharp_imgs)
    print(blur_imgs[0].shape)
    blur_imgs_eg = make_color(blur_imgs[0])
    # print()
    # blur_imgs = np.array(blur_imgs, dtype=np.uint8)
    # sharp_imgs = np.array(sharp_imgs, dtype=np.uint8)
    # np.savetxt('./image_blur', np.squeeze(blur_imgs_eg), fmt='%i')
    cv2.imshow('image_blur', blur_imgs[0])
    cv2.imshow('image_sharp', sharp_imgs[0])
    # cv2.imshow('img_eg', blur_imgs_eg)

    cv2.waitKey(0)
