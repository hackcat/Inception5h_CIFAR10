# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:32:17 2017

@author: notebook
"""
import sys
import pickle
import numpy as np
import os
import tensorflow as tf


def unpickle(file):
        fp = open(file, 'rb')
        if sys.version_info.major == 2:
            data = pickle.load(fp)
        elif sys.version_info.major == 3:
            data = pickle.load(fp, encoding='latin-1')
        fp.close()
        return data

def load_cifar10 (path):
    """ Load CIFAR_10 to X_train,y_train,X_test,y_test"""
    X_train=None
    y_train=[]
    for i in range(1,6):
        data_dic = unpickle(path+"cifar-10-batches-py/data_batch_{}".format(i))
        if i == 1:
            X_train = data_dic['data']
        else:
            X_train = np.vstack((X_train, data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(path+"cifar-10-batches-py/test_batch")
    X_test = test_data_dic['data']
    y_test = np.array(test_data_dic['labels'])
    shape=(-1,32,32,3)
    X_test = X_test.reshape(shape, order='F')
    X_train = X_train.reshape(shape, order='F')
    X_test = np.transpose(X_test, (0, 2, 1, 3))
    X_train = np.transpose(X_train, (0, 2, 1, 3))
    y_train = np.array(y_train)
    
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    
    return X_train,y_train,X_test,y_test



    
    
def data_to_tfrecord(images, labels, filename):
    """ Save data into TFRecord """
    print("Converting data into %s ..." % filename)
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(images):
        img_raw = img.tobytes()
        label = int(labels[index])
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()
    

    
    
def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    if is_train == True:
        # 1. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [24, 24, 3])
        # 2. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)
        # 3. Randomly change brightness.
        img = tf.image.random_brightness(img, max_delta=63)
        # 4. Randomly change contrast.
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        try: # TF12
            img = tf.image.per_image_standardization(img)
        except: #earlier TF versions
            img = tf.image.per_image_whitening(img)

    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        try: # TF12
            img = tf.image.per_image_standardization(img)
        except: #earlier TF versions
            img = tf.image.per_image_whitening(img)
    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label

def bottleneck_npy_cifar10(path,sess,tensor_name,t_input):
    X_train,y_train,X_test,y_test=load_cifar10(path)
    bottleneck=sess.graph.get_tensor_by_name(tensor_name)
    tensor_name=tensor_name.split('/')[-1]
    layer_name=tensor_name.split(':')[0]
    if not os.path.exists('y_train_'+layer_name+'.npy'):
        print('Converting y_train.......................')
        np.save('y_train_'+layer_name,y_train)
    if not os.path.exists('y_test_'+layer_name+'.npy'):
        print('Converting y_test.......................')
        np.save('y_test_'+layer_name,y_test)
    if not os.path.exists('X_train_'+layer_name+'.npy'):
        print('Converting X_train......................')
        X=[]
        for i in range(X_train.shape[0]):
            bot=sess.run(bottleneck,{t_input:np.expand_dims(X_train[i],0)})
            X.append(bot.squeeze())
        np.save('X_train_'+layer_name,np.array(X))
    if not os.path.exists('X_test_'+layer_name+'.npy'):
        print('Converting X_test......................')
        X=[]
        for i in range(X_test.shape[0]):
            bot=sess.run(bottleneck,{t_input:np.expand_dims(X_test[i],0)})
            X.append(bot.squeeze())
        np.save('X_test_'+layer_name,np.array(X))
    print('Complete!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
def load_bottleneck_npy (layer_name):
    """Return:X_train,y_train,X_test,y_test"""
    X_train=np.load('X_train_'+layer_name+'.npy')
    y_train=np.load('y_train_'+layer_name+'.npy')
    X_test=np.load('X_test_'+layer_name+'.npy')
    y_test=np.load('y_test_'+layer_name+'.npy')
    return X_train,y_train,X_test,y_test


    