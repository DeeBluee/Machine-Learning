import tensorflow as tf
import numpy as np
import os
"""
函数说明:
Parameters:
    file_dir:文件路径
Returns:
    image_list:图片列表
    label_list:标签列表
"""
def get_files(cat_file_dir, dog_file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(cat_file_dir):
        cats.append(cat_file_dir + file)
        label_cats.append(0)
    for file in os.listdir(dog_file_dir):
        dogs.append(dog_file_dir + file)
        label_dogs.append(1)
    # 打乱文件顺序
    image_list = np.hstack((cats, dogs))  # np.vstack 是将数据竖向排列，hstack是将数据横向排列
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)   # 乱序

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

"""
函数说明:生成相同大小的批次
Parameters:
    image:图像
    label:标签列表
    image_W、image_H:图像的宽高
    batch_size:每个batch有多少张图片
    capacity:队列容量
Returns:
    image_batch:图像
    label_batch
"""
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 将python.list转换成tf能够识别的格式
    image = tf.cast(image,tf.string)
    label = tf.cast(label, tf.int32)

    #生成队列
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.io.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    #统一图片方法
    # 视频方法
    # image = tf.image.resize_image_with_crop_or_pad(image, image)W, image_H)
    # 我的方法
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image) # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=65, capacity=capacity)
    print(image)
    return image_batch, label_batch




