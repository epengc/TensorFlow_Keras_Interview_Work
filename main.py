#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import,division,print_function,unicode_literals
from utils.util import *
from model.mobile_net import *
import tensorflow as tf
import pathlib 
import random
import IPython.display as display
import os
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


def demo_train_val():
    '''Demonstrate the training and validation process

    Input parameters are sent by using parse_parameters().
    
    Args:
        None

    Returns:
        A sorted list to tell demo_test() functon how many classes we have.
        example in this case:

        label_names=['cocacola','pepsi']
    '''
    args = parse_parameters()
    # get the path of training and validation dataset 
    data_root_orig = os.path.abspath(args.data_root)
    data_root = pathlib.Path(data_root_orig)
    all_images_paths = list(data_root.glob('*/*'))
    all_images_paths = [str(path) for path in all_images_paths]
    random.shuffle(all_images_paths)    # to get a random order of input for training
    image_count = len(all_images_paths)
    print('All images amount in training and validation is -- {}'.format(image_count))
    
    # assign labels to data
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names)) # in demo_train_val(), label_to_index={'cocacola':0, 'pepsi':1}
    all_images_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]
    
    # assign training samples and validation samples 
    index = int(args.ratio*image_count)
    train_ds = prepare_train_dataset(all_images_paths[0:index], all_images_labels[0:index], AUTOTUNE)
    val_ds = prepare_val_dataset(all_images_paths[index::], all_images_labels[index::], AUTOTUNE)
    
    # getting a model instance for training and validation
    model = mobile_net(classes=len(label_names))
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=["accuracy"])
    model.summary()
    steps_per_epoch = tf.math.ceil(index/args.train_batch_size).numpy() # calculate how many iterations within each epochs
    print('steps_per_epoch = {}'.format(steps_per_epoch))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=args.weights) # setting up the saving environment for *.pb 
    ]
    model.fit(train_ds, epochs=args.max_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=2)
    model.evaluate(val_ds, batch_size=args.val_batch_size, verbose=1)
    model.save(os.path.join(args.weights,'final_model.h5'))
    # convert model to *.tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model) 
    tflite_model = converter.convert()
    with tf.io.gfile.GFile(os.path.join(args.weights,'model.tflite'),'wb') as f:
        f.write(tflite_model)

    return label_names


def demo_test(label_names):
    '''Demonstrate the test process.

    10 test samples will be classified

    Args:
        label_names: a sorted list contraining the names of classes

    Returns:
        None
    '''
    args = parse_parameters()
    # getting the test data path
    data_test = os.path.abspath(args.data_test)
    data_test = pathlib.Path(data_test)
    test_images_paths = list(data_test.glob('*/*'))
    test_images_paths = [str(path) for path in test_images_paths]
    test_count = len(test_images_paths)
    print('All images amount in testing is -- {}'.format(test_count))

    # getting the test dataset
    test_ds = prepare_test_dataset(test_images_paths, AUTOTUNE)
    
    # loading the model from saved *.h5 file for testing
    model=tf.keras.models.load_model(os.path.join(args.weights,'final_model.h5'))
    predicts = model.predict(test_ds)
    predicts = tf.keras.backend.argmax(predicts, 1).numpy()

    # setting up the display format for showing the predictions 
    label_to_index = dict((index, name) for index, name in enumerate(label_names)) # label_to_index={'0':'cocacola', '1':'pepsi'}
    results = dict((name, label_to_index[predicts[index]]) for index, name in enumerate(test_images_paths))
    print('The predictions from cocacola and pepsi classifer are ------------------------------------------')
    for result in results:
        print('predictions: {} is {}'.format(result, results[result]))


if __name__ == '__main__':
    label_names = demo_train_val()
    demo_test(label_names)

