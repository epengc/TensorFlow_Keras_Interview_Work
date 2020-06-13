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

AUTOTUNE = tf.data.experimental.AUTOTUNE


def demo_train_val():

    args = parse_parameters()
    data_root_orig = os.path.abspath(args.data_root)
    data_root = pathlib.Path(data_root_orig)
    #print('Dataset path = {}'.format(data_root))

    #for item in data_root.iterdir():
    #    print(item)
    all_images_paths = list(data_root.glob('*/*'))
    #print('some_images_paths = {}'.format(all_images_paths[-5:]))
    all_images_paths = [str(path) for path in all_images_paths]
    random.shuffle(all_images_paths)
    image_count = len(all_images_paths)
    #print('image_count = {}'.format(image_count))

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    #print('label_name = {}'.format(label_names))
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    #print('label_to_index = {}'.format(label_to_index))
    
    #for path in all_images_paths:
    #    print(pathlib.Path(path).parent.name)
    all_images_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]
    #print(all_images_labels[:10])

    index = int(args.ratio*image_count)
    train_images_labels = all_images_labels[0:index]
    #print('train_data number is {}'.format(len(train_images_labels)))
    val_images_labels = all_images_labels[index::]
    #print('val_data number is {}'.format(len(val_images_labels)))

    train_ds = prepare_train_dataset(all_images_paths[0:index], all_images_labels[0:index], AUTOTUNE)
    val_ds = prepare_val_dataset(all_images_paths[index::], all_images_labels[index::], AUTOTUNE)

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
    
    model.fit(train_ds, epochs=args.max_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=2)
    model.evaluate(val_ds, batch_size=args.val_batch_size, verbose=1)
    model.save(args.weights+'/final_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(os.path.join(args.weights,'model.tflite'),'wb') as f:
        f.write(tflite_model)

    return label_names


def demo_test(label_names):
    # Preparing for test dataset
    args = parse_parameters()
    data_test = os.path.abspath(args.data_test)
    data_test = pathlib.Path(data_test)
    test_images_paths = list(data_test.glob('*/*'))
    print(test_images_paths)
    test_images_paths = [str(path) for path in test_images_paths]
    test_count = len(test_images_paths)
    print('The test data amount is ----------------- {}'.format(test_count))
    test_ds = prepare_test_dataset(test_images_paths, AUTOTUNE)

    label_to_index = dict((index, name) for index, name in enumerate(label_names))
    # crreate a new blank model
    model=tf.keras.models.load_model(args.weights+'/final_model.h5')
    predicts = model.predict(test_ds)
    predicts = tf.keras.backend.argmax(predicts, 1).numpy()
    print(predicts)
    results = dict((name, label_to_index[predicts[index]]) for index, name in enumerate(test_images_paths))

    for result in results:
        print('predictions: {} is {}'.format(result, results[result]))



if __name__ == '__main__':
    label_names = demo_train_val()
    demo_test(label_names=['cocacola', 'pepsi'])

