#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:05:06 2018

@author: kaygudo
"""

import tensorflow as tf
import numpy as np
import os


def Generator_Network(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, model_param=None,learn_scale=True):

    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        if model_param is None:
            WG_init = tf.truncated_normal_initializer()
            BG_init = tf.truncated_normal_initializer()
            scale_out_G_initializer = tf.constant_initializer(value=1.0)
            lstm_initializer = None
            bias_start = 1.0
        else:
            WG_init = tf.constant_initializer(value=model_param['generator/WG:0'])
            BG_init = tf.constant_initializer(value=model_param['generator/BG:0'])
            try:
                scale_out_G_initializer = tf.constant_initializer(value=model_param['generator/scale_out_G:0'])
            except KeyError:
                scale_out_G_initializer = tf.constant_initializer(value=1)
                assert learn_scale
            lstm_initializer = tf.constant_initializer(value=model_param['generator/rnn/lstm_cell/kernel:0'])

            bias_start = model_param['generator/rnn/lstm_cell/bias:0']

        WG = tf.get_variable(name='WG', shape=[hidden_units_g, num_generated_features], initializer=WG_init)
        BG = tf.get_variable(name='BG', shape=num_generated_features, initializer=BG_init)

        inputs = z

        cell=tf.nn.rnn_cell.LSTMCell(hidden_units_g, initializer=lstm_initializer,state_is_tuple=True,reuse=reuse)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn( cell=cell, dtype=tf.float32,
            sequence_length=[seq_length]*batch_size, inputs=inputs)
        
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, WG) + BG
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
        
    return output_3d



def Discriminator_Network(x, hidden_units_d, seq_length, batch_size, reuse=False, batch_mean=False):
    
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        WD = tf.get_variable(name='WD', shape=[hidden_units_d, 1],
                initializer=tf.truncated_normal_initializer())
        BD = tf.get_variable(name='BD', shape=1,
                initializer=tf.truncated_normal_initializer())

        inputs = x
        
        cell=tf.nn.rnn_cell.LSTMCell(hidden_units_d,state_is_tuple=True,reuse=reuse)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=inputs)
        
        einsum = tf.einsum('ijk,km', rnn_outputs, WD) + BD
        output = tf.nn.sigmoid(einsum)
        
    return output, einsum


def sample_Z(batch_size, seq_length, latent_dim):
    
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))

    return sample


def GAN_loss_solver(hidden_units_g, hidden_units_d, seq_len, batch_size, num_generated_features,Z, X,learning_rate):

    G_sample = Generator_Network(Z,hidden_units_g, seq_len, batch_size, num_generated_features)
    D_real, D_logit_real  = Discriminator_Network(X, hidden_units_d, seq_len, batch_size)
    D_fake, D_logit_fake = Discriminator_Network(G_sample, hidden_units_d, seq_len, batch_size, reuse=True)
    
    
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), 1)
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), 1)

    D_loss = D_loss_real + D_loss_fake

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), 1)
    
    discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

    D_loss_mean_over_batch = tf.reduce_mean(D_loss)
    D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss_mean_over_batch, var_list=discriminator_vars)

    G_loss_mean_over_batch = tf.reduce_mean(G_loss)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss_mean_over_batch, var_list=generator_vars)
    
    return D_loss, G_loss, D_solver, G_solver



def get_batch(samples, batch_size, batch_idx, labels=None):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    if labels is None:
        return samples[start_pos:end_pos], None

    else:
        if type(labels) == tuple: 
            assert len(labels) == 2
            return samples[start_pos:end_pos], labels[0][start_pos:end_pos], labels[1][start_pos:end_pos]
        else:
            assert type(labels) == np.ndarray
            return samples[start_pos:end_pos], labels[start_pos:end_pos]



def train_epoch(epoch, samples, labels, sess, Z, X, D_loss, G_loss, D_solver, G_solver, 
                batch_size, D_rounds, G_rounds, seq_length, 
                latent_dim, num_generated_features, cond_dim):

    for batch_idx in range(0, int(len(samples) / batch_size) - (D_rounds + (cond_dim > 0)*G_rounds), D_rounds + (cond_dim > 0)*G_rounds):
        # update the discriminator
        for d in range(D_rounds):
            X_mb, Y_mb = get_batch(samples, batch_size, batch_idx + d, labels)

            noise = sample_Z(batch_size, seq_length, latent_dim)
            
            _,D_loss_curr = sess.run([D_solver,D_loss], feed_dict={X: X_mb, Z: noise})

        # update the generator
        for g in range(G_rounds):

            _,G_loss_curr = sess.run([G_solver,G_loss],feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim)})
            

    D_loss_cal = np.mean(D_loss_curr)
    G_loss_cal = np.mean(G_loss_curr)
    
    return D_loss_cal, G_loss_cal


def Save_Parameters(identifier, sess):
    
    directory=os.path.isdir("parameters")

    if directory==False:
        path = "parameters"

        try:  
            os.makedirs(path)
        except OSError:  
        
            print ("Creation of the directory %s failed", path)
        else:  
            print ("Successfully created the directory %s", path)
        

    param_path = './parameters/' + identifier + '.npy'
    model_parameters = dict()
    for v in tf.trainable_variables():
        model_parameters[v.name] = sess.run(v)
    np.save(param_path, model_parameters)
    print('Recorded', len(model_parameters), 'parameters to', param_path)
    return True

def load_parameters(identifier):

    load_path = './parameters/' + identifier + '.npy'
    model_parameters = np.load(load_path).item()
    return model_parameters


def Generate_Samples(identifier, epoch, seq_length,latent_dim, num_samples, hidden_units_g, num_generated_features):

    model_parameters = load_parameters(identifier + '_' + str(epoch))

    noise = tf.placeholder(tf.float32, [num_samples, seq_length, latent_dim])

    Z_samples = sample_Z(num_samples, seq_length, latent_dim)

    G_samples = Generator_Network(noise, hidden_units_g, seq_length, num_samples, num_generated_features, 
                              reuse=True, model_parameters=model_parameters)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        gen_samples = sess.run(G_samples, feed_dict={noise: Z_samples})

    tf.reset_default_graph()
    return gen_samples

