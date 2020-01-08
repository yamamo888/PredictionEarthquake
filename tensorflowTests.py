# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:28:22 2019

@author: yu
"""

import numpy as np
import tensorflow as tf
import pdb

# [2,3]
weight = tf.constant([[20,4,7],[19,3,3]])

pdb.set_trace()

ind = tf.where(weight>5,tf.gather_nd(weight),tf.zeros(tf.shape[0]))
#positive = tf.gather_nd(weight,ind)


softs = tf.zeros([2,3])
#softs[ind] = positive

# ----

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print(sess.run(ind))
print("----")
print(sess.run(positive))
print("----")
print(sess.run(softs))

pdb.set_trace()