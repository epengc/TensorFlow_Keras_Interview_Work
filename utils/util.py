#!/usr/bin/env python
# coding=utf-8
import sys
import os
import tensorflow as tf
import numpy as np
import argparse

def parse_parameters():
    '''Parameters index

    All parameters settings are listed here.

    Args:
        None

    Returns:
        args: a parse_args instance.
    '''
    parser = argparse.ArgumentParser(description='Visual Defense Homework Net')

    parser.add_argument('--data_root', default='./data', type=str, metavar='DIR',
                        help='path to dataset for training and validation')
    parser.add_argument('--data_test', default='./test', type=str, metavar='DIR',
                        help='path to dataset for test')
    parser.add_argument('--train_batch_size', default=4, type=int, metavar='N', 
                        help='Batch_size when training')
    parser.add_argument('--val_batch_size', default=2, type=int, metavar='N', 
                        help='Batch_size when validation')
    parser.add_argument('--max_epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--steps_per_epoch', default=10, type=int, metavar='N', 
                        help='iterations per epochs')
    parser.add_argument('--weights', default='./weights/', type=str, 
                        help='weights_backup')
    parser.add_argument('--ratio', default=0.9, type=float, 
                        help='Ratio of training over validation')
    args = parser.parse_args()

    return args


def preprocess_img(image):
    '''Decode and normalize the image data by using TensorFlow 2.2
    png and jpeg are acceptable. keras.image.preprocessing.load_img is another choice
    '''
    image = tf.image.decode_png(image,channels=3) 
    image = tf.image.resize(image,[192,192], preserve_aspect_ratio=False) #resize images
    image /= 255.0 # normalization 
    return image


def load_and_preprocess_image(path):
    '''Load the image file using TensorFlow 2.2'''
    image = tf.io.read_file(path)
    return preprocess_img(image)


def prepare_train_dataset(images_paths, images_labels, num_parallel_calls):
    '''Prepare the dataset for training
    Training data can be orgnized in flip-up, random shuffle, cropped and batch_size 

    Args:
        images_paths: image paths str in list
        images_labels: class label for each image
        num_parallel_calls: backend buffer setting

    Return:
        train_ds: [batch_size, 192, 192, 3] TensorFlow Dataset instance
    '''
    args = parse_parameters()
    path_ds = tf.data.Dataset.from_tensor_slices(images_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=num_parallel_calls)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(images_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    train_ds = image_label_ds.shuffle(buffer_size=len(images_paths))
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(args.train_batch_size)
    train_ds = train_ds.prefetch(buffer_size=num_parallel_calls)
    return train_ds


def prepare_val_dataset(images_paths, images_labels, num_parallel_calls):
    '''Prepare the dataset for validation

    Args:
        images_paths: image paths str in list
        images_labels: class label for each image
        num_parallel_calls: backend buffer setting

    Return:
        val_ds: [batch_size, 192, 192, 3] TensorFlow Dataset instance
    '''
    args = parse_parameters()
    path_ds = tf.data.Dataset.from_tensor_slices(images_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=num_parallel_calls)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(images_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    val_ds = image_label_ds.batch(args.val_batch_size)
    return val_ds


def prepare_test_dataset(images_paths, num_parallel_calls):
    '''Prepare the dataset for test

    Args:
        images_paths: image paths str in list
        num_parallel_calls: backend buffer setting

    Return:
        test_ds: [batch_size, 192, 192, 3] TensorFlow Dataset instance
    '''
    args = parse_parameters()
    path_ds = tf.data.Dataset.from_tensor_slices(images_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=num_parallel_calls)
    val_ds = image_ds.batch(1)
    return val_ds
