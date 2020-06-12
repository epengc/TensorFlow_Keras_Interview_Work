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
    
if __name__ == '__main__':
    demo_basic()


