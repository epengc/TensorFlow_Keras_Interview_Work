#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

def mobile_net(classes=2):
    '''Kernal function of the classifier
    
    For CPU supported TensorFlow 2.2, using MobileNetV2 as backbone in classifier and loading
    its pretrained weights will make an easier, faster training and reliable for mobile transfer.
    (If the MobileNetV2 is inefficient, the customer model can be defined here.)

    Args:
        classes: int number to indicate amount of classes
    
    Returns:
        model: the subclass of keras.Model
    '''
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False) 
    mobile_net.trainable=False
    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(), 
        tf.keras.layers.Dense(classes, activation='softmax')
    ])
    return model
