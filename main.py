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

AUTOTUNE = tf.data.experimental.AUTOTUNE

def demo_basic():

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

if __name__ == '__main__':
    demo_basic()


