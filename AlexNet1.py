import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as  plt
import numpy as np
import os
import cv2


# 加载数据
basedir = './dataset/'
train_dir = os.path.join(basedir, 'train/')
val_dir = os.path.join(basedir, 'validation/')
train_cats_dir = os.path.join(train_dir, 'cats/')
train_dogs_dir = os.path.join(train_dir, 'dogs/')
val_cats_dir = os.path.join(val_dir, 'cats/')
val_dogs_dir = os.path.join(val_dir, 'dogs/')



img = cv2.imread(train_cats_dir+os.listdir(train_cats_dir)[1])

train_images = []
train_labels = []
for item in os.listdir(train_cats_dir):
    img = cv2.imread(train_cats_dir + item)
    try:
        img = cv2.resize(img, (224, 224))
    except:
        continue
    img = img / 255.0
    train_images.append(img)
    train_labels.append(0)



for item in os.listdir(train_dogs_dir):
    img = cv2.imread(train_cats_dir + item)
    try:
        img = cv2.resize(img, (224, 224))
    except:
        continue
    img = img / 255.0
    train_images.append(img)

    train_labels.append(1)
train_images = np.array(train_images)
train_labels = np.array(train_labels)
train_images.reshape(-1, 224, 224, 3)
print('data is ready!')

# 设置参数
batch_size = 128
IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs= 20

model = tf.keras.models.Sequential([
    # 第一个卷积层
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu',
                           input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    # 第一个参数为卷积核个数，第二个参数为卷积核尺寸，为(width, height)，如果两者相同，用一个数字即可
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                 strides=(2, 2),
                                 padding='valid'),

    # 第二个卷积层
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='valid'),

    # 第三-五卷积层
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='valid'),

    # 第六-八连接层

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # 输出

    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation('softmax')
])

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练
print('开始训练')
history = model.fit(train_images, train_labels,
                    steps_per_epoch= 27,
                    epochs=epochs,
                    )
# 可视化训练结果
# acc = history.history['accuracy']
# loss = history.history['loss']
#
# epochs = range(5)
# plt.figure()
# plt.plot(epochs, acc, color='b', label='train_acc' )
# plt.plot(epochs, loss, color='r', label='train_loss')
# plt.legend()
# plt.show()


# 预测
# predictions = model.predict(test_images)
