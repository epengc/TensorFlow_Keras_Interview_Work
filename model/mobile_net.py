#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

def mobile_net(classes=2):
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False) 
    mobile_net.trainable=False
    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(), 
        tf.keras.layers.Dense(classes, activation='softmax')
    ])
    return model
