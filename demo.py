import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

"""
本教程遵循一个基本的机器学习工作流程:
-检查和理解数据
-建立输入管道
-建立模型
-训练模型
-测试模型
-改进模型并重复该过程
"""

base_dir = './dataset/'
train_dir = os.path.join(base_dir, 'train/')
validation_dir = os.path.join(base_dir, 'validation/')
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

"""
    为方便起见，设置预处理数据集和训练网络时要使用的变量。
"""
batch_size = 128
epochs = 30
IMG_HEIGHT = 227
IMG_WIDTH = 227

num_cats_tr = len(os.listdir(train_cats_dir))  # total training cat images: 1000
num_dogs_tr = len(os.listdir(train_dogs_dir))  # total training dog images: 1000

num_cats_val = len(os.listdir(validation_cats_dir))  # total validation cat images: 500
num_dogs_val = len(os.listdir(validation_dogs_dir))  # total validation dog images: 500

total_train = num_cats_tr + num_dogs_tr  # Total training images: 2000
total_val = num_cats_val + num_dogs_val  # Total validation images: 1000


# 该函数将图像绘制成1行5列的网格形式，图像放置在每一列中。
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


"""
数据准备
    将图像格式化成经过适当预处理的浮点张量，然后输入网络:
    - 从磁盘读取图像。
    - 解码这些图像的内容，并根据它们的RGB内容将其转换成适当的网格格式。
    - 把它们转换成浮点张量。
    - 将张量从0到255之间的值重新缩放到0到3
    1之间的值，因为神经网络更喜欢处理小的输入值。
    幸运的是，所有这些任务都可以用tf.keras提供的ImageDataGenerator类来完成。
    它可以从磁盘读取图像，并将它们预处理成适当的张量。它还将设置发生器，将这些图像转换成一批张量——这对训练网络很有帮助。
"""

image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  #
    rotation_range=45,   # 旋转范围
    width_shift_range=.15,  # 水平平移范围
    height_shift_range=.15,  # 垂直平移范围
    horizontal_flip=True,   # 水平翻转
    zoom_range=0.5       # 缩放范围
)

# 在为训练和验证图像定义生成器之后，flow_from_directory方法从磁盘加载图像，应用重新缩放，并将图像调整到所需的尺寸。
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,  # batch数据大小
                                                     directory=train_dir,   # 目标文件夹
                                                     shuffle=True,  # 是否打乱数据
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),  # resize图像(height, width)
                                                     class_mode='binary')  # 返回1D的二值标签

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# 创建验证集数据生成器
image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')

# 创建模型
model_new = tf.keras.models.Sequential([

    # 卷积函数
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # 池化函数
    tf.keras.layers.MaxPooling2D(),
    # 按概率丢弃神经网络单元，防止过拟合
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    # 卷积层无法连接Dense全连接层，需要将数据通过Flatten压平
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
])
#
# # 编译模型
# # 这边选择ADAM优化器和二进制交叉熵损失函数。要查看每个训练时期的训练和验证准确性，请传递metrics参数。
model_new.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model_new.summary()

# 在成功地将数据扩充引入训练样例并向网络中添加Dropout后，训练此新网络:
history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# 可视化模型
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')


plt.show()