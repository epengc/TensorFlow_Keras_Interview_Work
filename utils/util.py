#!/usr/bin/env python
# coding=utf-8
import sys
import os
import tensorflow as tf
import numpy as np
import argparse

def parse_parameters():
    parser = argparse.ArgumentParser(description='Visual Defense Homework Net')
    parser.add_argument('--data_root', default='./data', type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--batch_size', default=4, type=int, metavar='N', 
                        help='Batch_size when training')
    parser.add_argument('--max_epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--steps_per_epoch', default=10, type=int, metavar='N', 
                        help='iterations per epochs')
    parser.add_argument('--weights', default='./weights/', type=str, 
                        help='weights_backup')
    args = parser.parse_args()
    return args

def preprocess_img(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image,[192,192]) #resize images
    image /= 255.0 # normalization 
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_img(image)

