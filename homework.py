#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
import pathlib 
import random
import IPython.display as display
import os
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE
data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos',untar=True)

data_root = pathlib.Path(data_root_orig)
print(data_root)

for item in data_root.iterdir():
    print(item)


all_images_paths = list(data_root.glob('*/*')) #获取所有文件路径
print(all_images_paths[-5:])
all_images_paths = [str(path) for path in all_images_paths] #将文件路径传入列表
random.shuffle(all_images_paths) #打乱文件路径顺序
image_count = len(all_images_paths) #查看文件数量
print(image_count)

for n in range(3):
    image_path = random.choice(all_images_paths) #随机选择
    display.display(display.Image(image_path)) #图片显示

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)

label_to_index = dict((name,index)for index,name in enumerate(label_names)) #转数字
print(label_to_index)

for path in all_images_paths:
    print(pathlib.Path(path).parent.name) #上级路径

all_images_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]
print(all_images_labels[:10]) #显示前10个图片标签


def preprocess_img(image):
    image = tf.image.decode_jpeg(image,channels=3) #映射为图片
    image = tf.image.resize(image,[192,192]) #修改大小
    image /= 255.0 #归一化
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path) #这里注意的是这里读到的是许多图片参数
    return preprocess_img(image)

path_ds = tf.data.Dataset.from_tensor_slices(all_images_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE) #通过路径加载图片数据集
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_images_labels,tf.int64)) #读入标签
image_label_ds = tf.data.Dataset.zip((image_ds,label_ds)) #图片和标签整合

Batch_size = 32
ds = image_label_ds.shuffle(buffer_size=image_count) #打乱数据
ds = ds.repeat() #数据重复
ds = ds.batch(Batch_size) #分割batch
ds = ds.prefetch(buffer_size=AUTOTUNE) #使数据集在后台取得 batch

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False) 
mobile_net.trainable=False

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(), #平均池化
    tf.keras.layers.Dense(len(label_names),activation='softmax') #分类
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.summary()

steps_per_epoch=tf.math.ceil(len(all_images_paths)/Batch_size).numpy()
print(steps_per_epoch)

model.fit(ds, epochs=10, steps_per_epoch=30,verbose=2)
