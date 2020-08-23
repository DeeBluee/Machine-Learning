import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as  plt
import numpy as np
# 加载数据
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# 建立标签与名字的映射
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print(train_images.shape)  (60000, 28, 28)
# print(train_labels.shape)  (60000,)
# print(test_images.shape)  (10000, 28, 28)
# print(test_labels.shape)  (10000,)


# 处理数据

# 图片展示
# plt.figure()  # 创建画布
# plt.imshow(train_images[1])   # 将某图片放在画布上
# plt.colorbar()  # 加入颜色条
# plt.grid(False)  # 关闭网格
# plt.show()  # 显示画布

train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# 构造网络
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ]
)

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 训练与验证
history = model.fit(train_images, train_labels, epochs=5)

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
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))   # 返回np数组中最大值的索引值

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks(())
    plt.yticks(())

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{}{:2.0f}%({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

img = test_images[0]

img = (np.expand_dims(img,0))

print(img.shape)
predictions_single = model.predict(img)

print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()


