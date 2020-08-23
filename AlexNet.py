import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



basedir = './dataset/'
train_dir = os.path.join(basedir, 'train/')
val_dir = os.path.join(basedir, 'validation/')
train_cats_dir = os.path.join(train_dir, 'cats/')
train_dogs_dir = os.path.join(train_dir, 'dogs/')
val_cats_dir = os.path.join(val_dir, 'cats/')
val_dogs_dir = os.path.join(val_dir, 'dogs/')
total_train = len(os.listdir(train_dir))
total_val = len(os.listdir(val_dir))




# 该函数将图像绘制成1行5列的网格形式，图像放置在每一列中。
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


train_image_generate = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)
batch_size = 128
IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 10

train_data_generate = train_image_generate.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)



val_image_generate = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)


val_data_generate = val_image_generate.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')





model = tf.keras.models.Sequential([
# 第一个卷积层
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu',
                           input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    # 第一个参数为卷积核个数，第二个参数为卷积核尺寸，为(width, height)，如果两者相同，用一个数字即可
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                 strides=(2, 2),
                                 padding='valid'),

    #第二个卷积层
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
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# categorical_crossentropy与softmax匹配使用
#sigmoid 与二进制交叉熵损失函数BinaryCrossentropy(from_logits=True)
# 因为模型最后一层的sotfmax已经概率化输出了，所以from_logits=False，如果输出最后结果没有
#进行激活函数sigmoid的映射之类的就设置成True
model.summary()


history = model.fit_generator(
    train_data_generate,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_generate,
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