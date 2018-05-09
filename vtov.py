# -*- coding: utf-8 -*-
"""
Created on Thu May  3 21:46:34 2018

@author: yu
"""
import pdb
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.examples.tutorials.mnist import input_data

dataPath = './CycleBranch2'


#dataset
def data(fname):

    fullpath = os.path.join(dataPath,fname)
    print('reading',fullpath)
    data = open(fullpath).readlines()
    
    return data

#list to array
def data2array(data):
    listv = []
    for i in np.arange(len(data)):
        dataset = data[i].split(',')
        listv.append(dataset)
    
    
    list2 = []
    #listyear = []
    #listch_amount = []
    for j in np.arange(len(listv)):
        listnum = np.array(listv)
        list2.append(listnum[j][:])
        #listyear.append(listnum[j][1])#array[1]:年
        #listch_amount(listnum[j][2])#array[2]:変化量
    
    list2num = np.array(list2).astype(np.float32)#[3765,10] 
    #pdb.set_trace()
    return list2num
    #return flat_v

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name)

def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name)

#def conv2d(inputs,w,b,strides):
#   return tf.nn.relu(tf.nn.conv2d(inputs,w,strides=strides,padding='SAME')+b)

def conv1d(inputs,w,b,stride):
    return tf.nn.relu(tf.nn.conv1d(inputs,w,stride,padding='SAME')+b)

def conv2d_layers_transpose(inputs):
    return tf.layers.conv2d_transpose(inputs,filters=10,kernel_size=2,padding='same',strides=2)

#def conv1d_transpose(inputs, w, b, output_shape, stride):
#	 return tf.nn.relu(nn_ops.conv1d_transpose(inputs, w, output_shape=output_shape, stride=stride, padding='SAME') + b)
	 
#def max_pool2d(inputs):
#    return tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def max_pool1d(inputs):#size半分にpooling
    return tf.nn.pool(inputs,[2],'MAX','SAME',strides=[2])

####################encoder###################    
def encoder(x):
    
    #第1層
    convW1 = weight_variable([10,10,32],'convW1')
    convB1 = bias_variable([32],'convB1')
    conv1 = conv1d(liner_v,convW1,convB1,stride=3)#(1,30,32)
    
    pdb.set_trace()
    pool1 = max_pool1d(conv1)
    
    #第２層
    convW2 = weight_variable([10,32,64],'convW2')
    convB2 = bias_variable([64],'convB2')
    conv2 = conv1d(pool1,convW2,convB2,stride=2)
    
    pdb.set_trace()
    pool2 = max_pool1d(conv2)
    
    return pool2

###################decoder#########################
def decoder(z):
    
    pdb.set_trace()
    convW1 = weight_variable([10,64,64],'convW1')
    convB1 = bias_variable([64],'convB1')
    conv1 = conv1d(z,convW1,convB1,stride=2)#[1,2,64]
    
    pdb.set_trace()
    
    #conv1d_transposeが使えないので３次元にreshape
    conv1to2 = tf.reshape(conv1,[-1,1,2,64])#[1,1,1,32]
    
    upsamples1 = conv2d_layers_transpose(conv1to2)
    upsamples2 = conv2d_layers_transpose(upsamples1)
    
    #2次元にreshape
    upsamples2to1 = tf.reshape(upsamples2,[1,-1,32])
    pdb.set_trace()
    
    convW2 = weight_variable([16,32,32],'convW2')
    convB2 = bias_variable([32],'convB2')
    conv2 = conv1d(upsamples2to1,convW2,convB2,stride=5)
    
    pdb.set_trace()
    
    convW3 = weight_variable([1,32,16],'convW3')
    convB3 = bias_variable([1],'convB3')
    output = conv1d(conv2,convW3,convB3,stride=5)
    
    return output


data = data('shortlogv.txt')
data_v = data2array(data)
#[バッチ数、縦、横]:60*1にreshape
liner_v = tf.reshape(data_v,[-1,60,10])
#入力：ｘ
x = tf.placeholder(tf.float32,shape=[None,600])#shortlogv[60*10]

#pdb.set_trace()
train_z = encoder(liner_v)#[]

#pdb.set_trace()
train_xr = decoder(train_z)