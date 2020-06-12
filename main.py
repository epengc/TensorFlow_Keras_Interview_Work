#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
import pathlib 
import random
import IPython.display as display
import os
import matplotlib.pyplot as plt
from utils.util import *
from model.mobile_net import *

def demo_basic():

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    args = parse_parameters()
    data_root_orig = os.path.abspath(args.data_root)
    data_root = pathlib.Path(data_root_orig)
    print('Dataset path = {}'.format(data_root))

    for item in data_root.iterdir():
        print(item)
    all_images_paths = list(data_root.glob('*/*'))
    print('some_images_paths = {}'.format(all_images_paths[-5:]))
    all_images_paths = [str(path) for path in all_images_paths]
    random.shuffle(all_images_paths)
    image_count = len(all_images_paths)
    print('image_count = {}'.format(image_count))

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print('label_name = {}'.format(label_names))
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print('label_to_index = {}'.format(label_to_index))
    
    for path in all_images_paths:
        print(pathlib.Path(path).parent.name)
    all_images_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]
    print(all_images_labels[:10])

    path_ds = tf.data.Dataset.from_tensor_slices(all_images_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_images_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    batch_size = args.batch_size
    ds = image_label_ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    model = mobile_net(classes=len(label_names))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=args.weights)
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=["accuracy"])
    model.summary()

    steps_per_epoch = tf.math.ceil(len(all_images_paths)/batch_size).numpy()
    print('steps_per_epoch = {}'.format(steps_per_epoch))
    
    model.fit(ds, epochs=args.max_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=2)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(os.path.join(args.weights,'model.tflite'),'wb') as f:
        f.write(tflite_model)
    
if __name__ == '__main__':
    demo_basic()


