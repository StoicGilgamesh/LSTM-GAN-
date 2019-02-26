# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:20:49 2018

@author: CHHRGUP
"""


import tensorflow as tf
import numpy as np
from math import ceil
from sklearn import preprocessing
import model_recurrentgan_tf
from time import time

split=0.9 #split ratio of Training, Test and Evalutation
seq_len=2 # Sequence Length
batch_size =34 # Batch Size
hidden_units_g=128 #number of hidden layers in Generator
hidden_units_d=128 # number of hidden layers in Discriminator
latent_dim=5 
num_signals=1
num_generated_features=1
learning_rate=0.0002 # learning rate
num_epochs=1000 #number of Epochs
D_rounds=5 # Discriminator Rounds
G_rounds=3 # Generator Rounds
identifier='faulty_gan' # Name for saving the parameter names
    
def split(samples, proportions, normalise=False, scale=False, labels=None, random_seed=None):
    """
    Split train/validation/test.
    """
    assert np.sum(proportions) == 1
    n_total = samples.shape[0]
    n_train = ceil(n_total*proportions[0])
    n_test = ceil(n_total*proportions[2])
    n_vali = n_total - (n_train + n_test)

    train_indices= samples[:n_train]
    train=train_indices

    vali_indices = samples[n_train:(n_train + n_vali)]
    vali=vali_indices

    test_indices = samples[(n_train + n_vali):]
    test=test_indices
    
    train = np.reshape(train,(train.shape[0],train.shape[1],1))
    test = np.reshape(test,(test.shape[0],test.shape[1],1))
    vali = np.reshape(vali,(vali.shape[0],vali.shape[1],1))
    
    if labels is None:
        return train, vali, test
    else:
        print('Splitting labels...')
        if type(labels) == np.ndarray:
            train_labels = labels[train_indices]
            vali_labels = labels[vali_indices]
            test_labels = labels[test_indices]
            labels_split = [train_labels, vali_labels, test_labels]
        elif type(labels) == dict:
            # more than one set of labels!  (weird case)
            labels_split = dict()
            for (label_name, label_set) in labels.items():
                train_labels = label_set[train_indices]
                vali_labels = label_set[vali_indices]
                test_labels = label_set[test_indices]
                labels_split[label_name] = [train_labels, vali_labels, test_labels]
        else:
            raise ValueError(type(labels))
        return train, vali, test, labels_split
    

def load_data():

    data = np.load('sp500_2.npy') # Name of the data in npy format
    
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    labels=None
    
    return data,labels,min_max_scaler


samples, labels, min_max_scaler =  load_data()

num_samples= samples.shape[0]
        
train, vali, test = split(samples, [0.8, 0.1, 0.1])
train = np.reshape(train,(train.shape[0],train.shape[1],1))
test = np.reshape(test,(test.shape[0],test.shape[1],1))
vali = np.reshape(vali,(vali.shape[0],vali.shape[1],1))

train_labels, vali_labels, test_labels = None, None, None


labels = dict()
labels['train'], labels['vali'], labels['test'] = train_labels, vali_labels, test_labels

samples = dict()
samples['train'], samples['vali'], samples['test'] = train, vali, test


Z = tf.placeholder(tf.float32, [batch_size, seq_len, latent_dim])
X = tf.placeholder(tf.float32, [batch_size, seq_len, num_generated_features])

D_loss, G_loss, D_solver, G_solver = model_recurrentgan_tf.GAN_loss_solver(hidden_units_g, hidden_units_d, seq_len, batch_size, num_generated_features,Z, X,learning_rate)
G_sample = model_recurrentgan_tf.Generator_Network(Z,hidden_units_g, seq_len, batch_size, num_generated_features, reuse=True)


vis_freq = max(14000//num_samples, 1)
vis_Z = model_recurrentgan_tf.sample_Z(batch_size, seq_len, latent_dim)


init = tf.global_variables_initializer()

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

t0 = time()
best_epoch = 0
gen_waves=[]
dloss=[]
gloss=[]
cond_dim=0


print('epoch\ttime\tD_loss\tG_loss')
for epoch in range(num_epochs):
    D_loss_curr, G_loss_curr = model_recurrentgan_tf.train_epoch(epoch, samples['train'], labels['train'],
                                        sess, Z, X,D_loss, G_loss,D_solver, G_solver, 
                                        batch_size, D_rounds, G_rounds, seq_len, 
                                        latent_dim, num_generated_features, cond_dim)

    if epoch> 199:

        vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
        gen_waves.append(vis_sample)
        
    t = time() - t0

   
    dloss.append(D_loss_curr) 
    gloss.append(G_loss_curr)


    print('%d\t%.2f\t%.4f\t%.4f\t' % (epoch, t, D_loss_curr, G_loss_curr))

    
    if epoch % 50 == 0:
        model_recurrentgan_tf.Save_Parameters(identifier + '_' + str(epoch), sess)

        
gen_waves=np.asarray(gen_waves)
np.save('genwaves.npy',gen_waves)
np.save('D_LOSS.npy',dloss)
np.save('G_LOSS.npy',gloss)

model_recurrentgan_tf.Save_Parameters(identifier + '_' + str(epoch), sess)


a=np.load('genwaves.npy')
for i in range(epoch-1):
    a=a[i,:,0]
    unscaled=min_max_scaler.inverse_transform(a)
    np.save('genwaves_transf'+ str(i),unscaled)
