# Visual Defence Homework -- Cocacola, Pepsi can classifier
Cocacola, pepsi can classifier based on MobileNetV2 for Keras and Tensorflow 2.2
## Datasets for trainig and validation
./data/cocacola 74 png  
./data/pepsi 63 png  
124 png are selected for training  
 13 png are selected for validation  
## Dataset for test
./test/test test_00.png, test_01.png, ... test_09.png  
10 png are selected for test  
## Architectures
'''python
tf.keras.Sequential([
        MobileNetV2(input_shape=(192,192,3), include_top=False),
        tf.keras.layers.GlobalAveragePooling2D(), 
        tf.keras.layers.Dense(classes, activation='softmax')
    ])
'''
