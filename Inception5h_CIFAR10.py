# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:20:54 2017

@author: notebook
"""

import os
import urllib.request
import zipfile
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from sklearn.cross_validation import train_test_split


from preprocessing import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

url =  'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
model_zip = os.path.split(url)[-1]
cwd=os.getcwd()
model_dir='Inception5h'
model_pb='tensorflow_inception_graph.pb'
local_zip=os.path.join(cwd,model_dir,model_zip)
local_pb=os.path.join(cwd,model_dir,model_pb)
if not os.path.exists(local_pb):
    print('Not having tensorflow_inception_graph.pb locally')
    with urllib.request.urlopen(url) as response, open(local_zip, 'wb') as out_file:
        print('Downloading Inception5h.zip..............................')
        data = response.read()
        out_file.write(data)
    with zipfile.ZipFile(local_zip,'r') as zip_ref:
        print ('Unzipping...............................')
        zip_ref.extractall(os.path.join(cwd,model_dir))
        print ('Done!!!!!!!!!!!!!!!!!!!!')
else:
    print ('Already having tensorflow_inception_graph.pb locally!!!!!!!!!!!')


graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

with tf.gfile.FastGFile(local_pb, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

#for node in graph_def.node:
#    print(node.name)
    
t_input = tf.placeholder(tf.float32, name='input') 
tf.import_graph_def(graph_def, {'input':t_input})
#nodes=[op.name for op in graph.get_operations()]    

## To inspect the architecture of this model,I write an Summary and lauch TensorBoard
#writer=tf.summary.FileWriter(logdir=os.path.join(cwd,'Eventfile'),graph=graph)
#random_summary=tf.summary.histogram('Input',graph.get_tensor_by_name('import/conv2d0_w:0'))
#summary=sess.run(random_summary,feed_dict={t_input:pic})
#writer.add_summary(summary)

tensor_name='import/maxpool4:0'  
layer_name=tensor_name.split('/')[-1].split(':')[0]     

bottleneck_npy_cifar10('../',sess,tensor_name,t_input)

X_train,y_train,X_test,y_test=load_bottleneck_npy(layer_name)

sess.close()
## Now let's build our model

sess = tf.InteractiveSession()    
X=tf.placeholder(tf.float32,shape=[None]+list(X_train.shape[1:]),name='X')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')
global_step=tf.Variable(0,trainable=False,name='global_step')
learning_rate=tf.placeholder(tf.float32,name='learning_rate')


network=tl.layers.InputLayer(X,name='Input_Layer')
network=tl.layers.FlattenLayer(network,name='Flatten_Layer')
network=tl.layers.DropoutLayer(network,keep=0.5,name='Drop_1')
network=tl.layers.DenseLayer(network,512,act=tf.nn.relu,W_init=tf.contrib.layers.xavier_initializer(),
                             b_init=tf.constant_initializer(0.5),name='FC_1')
network=tl.layers.DropoutLayer(network,keep=0.5,name='Drop_2')
network=tl.layers.DenseLayer(network,10,act=tf.identity,W_init=tf.contrib.layers.xavier_initializer(),
                             b_init=tf.constant_initializer(0.5),name='Output_Layer')

## Define Loss
logits=network.outputs
y_pre=tf.cast(tf.arg_max(tf.nn.softmax(logits),1),tf.int32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(y_pre,y_),tf.float32))
accuracy_sum=tf.summary.scalar('Accuracy',accuracy)

loss_data=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y_))
loss_sum=tf.summary.scalar('Loss',loss_data)

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_data,global_step=global_step)
writer_train=tf.summary.FileWriter('Eventfile/'+layer_name+'/train',graph=sess.graph)
writer_val=tf.summary.FileWriter('Eventfile/'+layer_name+'/val',graph=sess.graph)

## Training
tl.layers.initialize_global_variables(sess)

#%%
val_ratio=0.2
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=val_ratio)
#%%
epoch=5
batch_size=128

#%%
for i in range(epoch):
    j=0
    for X_train_a,y_train_a in tl.iterate.minibatches(X_train,y_train,batch_size,shuffle=True):
        feed_dict={X:X_train_a,y_:y_train_a,learning_rate:1e-5}
        feed_dict.update(network.all_drop)
        _,losum,loss=sess.run([optimizer,loss_sum,loss_data],feed_dict=feed_dict)
        if j%20 == 0:
            feed_dict={X:X_train_a,y_:y_train_a}
            feed_dict.update(tl.utils.dict_to_one(network.all_drop))
            acc,accsum=sess.run([accuracy,accuracy_sum],feed_dict)
            print('At epoch %d step %d,loss:%.5f \n training_batch %.5f' % (i+1,j+1,loss,acc))
            writer_train.add_summary(losum,sess.run(global_step))
            writer_train.add_summary(accsum,sess.run(global_step))
        j+=1
    # Validation
    acc_val=[]
    for X_val_a,y_val_a in tl.iterate.minibatches(X_val,y_val,batch_size,shuffle=False):
        feed_dict={X:X_val_a,y_:y_val_a}
        feed_dict.update(tl.utils.dict_to_one(network.all_drop))
        pre,acc=sess.run([y_pre,accuracy],feed_dict=feed_dict)
        acc_val.append(acc)
    acc_val_mean=np.mean(acc_val)
    print ('After epoch %d , val accuracy : %.5f' %(i+1,acc_val_mean))
    print(pre)
    summary = tf.Summary()
    summary.value.add(tag="Accuracy", simple_value=float(acc_val_mean))
    writer_val.add_summary(summary,sess.run(global_step))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    