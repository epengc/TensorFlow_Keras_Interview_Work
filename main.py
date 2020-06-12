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

    index = int(args.ratio*image_count)
    train_images_labels = all_images_labels[0:index]
    print('train_data number is {}'.format(len(train_images_labels)))
    val_images_labels = all_images_labels[index::]
    print('val_data number is {}'.format(len(val_images_labels)))

    train_ds = prepare_dataset(all_images_paths[0:index], all_images_labels[0:index], AUTOTUNE)
    val_ds = prepare_dataset(all_images_paths[index::], all_images_labels[index::], AUTOTUNE)
    ds = train_ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(args.train_batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    model = mobile_net(classes=len(label_names))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=args.weights)
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=["accuracy"])
    model.summary()

    steps_per_epoch = tf.math.ceil(index/args.train_batch_size).numpy()
    print('steps_per_epoch = {}'.format(steps_per_epoch))
    
    model.fit(ds, epochs=args.max_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=2)
    model.evaluate(val_ds.batch(2), batch_size=2, verbose=2)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(os.path.join(args.weights,'model.tflite'),'wb') as f:
        f.write(tflite_model)
    
if __name__ == '__main__':
    demo_basic()


