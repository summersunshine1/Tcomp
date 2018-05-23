#dense ffm train,slowly

import os
import sys
import tensorflow as tf
import numpy as np
import math
import pandas as pd

from sklearn.metrics import auc,log_loss,roc_auc_score
from getPath import *
pardir = getparentdir()
from commonLib import *
contest_dir = pardir + "/preliminary_contest_data/"

class FFM(object):
    """
    Field-aware Factorization Machine
    """
    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = config['k']
        # num of fields
        self.f = config['f']
        
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']
        self.feature2field = config['feature2field']
        self.p = len(feature2field)


    def add_placeholders(self):
        self.X = tf.placeholder('float32', [None,self.p])
        self.y = tf.placeholder('int64', [None,])
        self.keep_prob = tf.placeholder('float32')

    def inference(self):
        """
        forward propagation
        :return: labels for each sample
        """
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.matmul(self.X, w1), b)

        with tf.variable_scope('field_aware_interaction_layer'):
            v = tf.get_variable('v', shape=[self.p,self.k], dtype='float32',
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.field_aware_interaction_terms = tf.constant(0, dtype='float32')
            # build dict to find f, key of feature,value of field
            self.field_aware_interaction_terms = tf.multiply(0.5,
                                             tf.reduce_mean(
                                                 tf.subtract(
                                                     tf.pow(tf.matmul(self.X, v), 2),
                                                     tf.matmul(tf.pow(self.X, 2), tf.pow(v, 2))),
                                                 1, keep_dims=True))
            # for i in range(self.p):
                # for j in range(i+1,self.p):
                    # self.field_aware_interaction_terms += tf.multiply(
                        # tf.reduce_sum(tf.multiply(v[i,self.feature2field[i]], v[j,self.feature2field[j]])),
                        # tf.multiply(self.X[:,i], self.X[:,j])
                    # )
        # shape of [None, 2]
        self.y_out = tf.add(self.linear_terms, self.field_aware_interaction_terms)
        self.y_out_prob = tf.reduce_max(tf.nn.softmax(self.y_out),1)

    def add_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(model.y_out,1), tf.int64), model.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # add summary to accuracy
        tf.summary.scalar('accuracy', self.accuracy) 
        
    def add_auc(self):
        auc = tf.metrics.auc(self.y, self.y_out_prob)
        self.auc = auc

    def train(self):
        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False,dtype = 'int64')
        # define optimizer
        optimizer = tf.train.AdagradDAOptimizer(self.lr, global_step=self.global_step)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        """build graph for model"""
        print("add_placeholder...")
        self.add_placeholders()
        print("add_inference...")
        self.inference()
        print("add_loss...")
        self.add_loss()
        print("add_auc...")
        # self.add_auc()
        # print("add_train...")
        self.train()

def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state("checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        # logging.info("Loading parameters for the my CNN architectures...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("no model")
        # logging.info("Initializing fresh parameters for the my Factorization Machine")

def train_model(sess, model, train_path, validata_path, featurelength,epochs=10, print_every=5):
    """training model"""
    # Merge all the summaries and write them out to train_logs
    # merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter('train_logs', sess.graph)
    features = featurelength
    for e in range(epochs):
        num_samples = 0
        losses = []
        # get training data, iterable
        train_data = pd.read_csv(train_path, chunksize=model.batch_size,header = None,index_col=False)
        # batch_size data
        for data in train_data:
            actual_batch_size = len(data)
            samples = np.array(data)
            arr = np.array([one_hot_representation(sample[0], model.p) for sample in samples])
            batch_X = arr[:,:-1]
            batch_y = arr[:,-1]
            print(np.shape(batch_X))
            print(np.shape(batch_y))
            # create a feed dictionary for this batch
            _,global_step = sess.run([model.train_op,model.global_step], feed_dict={model.X: batch_X, model.y: batch_y, model.keep_prob:1})
            if global_step % print_every == 0:
                validata_model(sess, model, validata_path)
                saver.save(sess, "checkpoints/model", global_step=global_step)
        # print loss of one epoch
        # total_loss = np.sum(losses)/num_samples
        # print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss, e+1))
        
def validata_model(sess, model, validata_path):
    test_data = pd.read_csv(validata_path, chunksize = model.batch_size)
    true_y = []
    pred_y = []
    i = 0
    for data in test_data:
        actual_batch_size = len(data)
        samples = np.array(data)
        arr = np.array([one_hot_representation(sample[0], model.p) for sample in samples])
        true_y += list(arr[:,-1])
        # create a feed dictionary for this batch
        predict_y= sess.run([model.y_out_prob], feed_dict={model.X: arr[:,:-1]})
        pred_y += list(predict_y[0])
        i+=actual_batch_size
        
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)
    aucscore = roc_auc_score(true_y, pred_y)
    loss = log_loss(true_y, pred_y)
    print("validate loss :" + str(loss) + " validate auc : " + str(aucscore) +'\n')
    
        
def test_model(sess, model, test_path, print_every = 50):
    """training model"""
    # get testing data, iterable
    test_data = pd.read_csv(test_path,
                            chunksize=model.batch_size)
    test_step = 1
    # batch_size data
    for data in test_data:
        actual_batch_size = len(data)
        batch_X = []
        for i in range(actual_batch_size):
            sample = data.iloc[i,:]
            array = one_hot_representation(sample,model.features)
            batch_X.append(array)

        batch_X = np.array(batch_X)

        # create a feed dictionary for this batch
        feed_dict = {model.X: batch_X, model.keep_prob:1}
        # shape of [None,2]
        y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
        # write to csv files
        data['click'] = y_out_prob[0][:,-1]
        if test_step == 1:
            data[['id','click']].to_csv('FM_FTRL_v1.csv', mode='a', index=False, header=True)
        else:
            data[['id','click']].to_csv('FM_FTRL_v1.csv', mode='a', index=False, header=False)

        test_step += 1
        if test_step % 50 == 0:
            logging.info("Iteration {0} has finished".format(test_step))
            
if __name__=="__main__":
    config = {}
    config['lr'] = 0.01
    config['batch_size'] = 1000
    config['reg_l1'] = 2e-3
    config['reg_l2'] = 0
    config['k'] = 4
    config['f'] = 30
    
    feature2field = read_dic(contest_dir+'/ffm/featureindex_field_dic')
    config['feature2field'] = feature2field
    # get feature length
    feature_length = len(feature2field)
    # initialize FFM model
    model = FFM(config)
    # build graph for model
    train_path = contest_dir+'/gbm/ffmtrain'
    valid_path = contest_dir+'/gbm/ffmvalid'
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=5)
    
    with tf.Session(config=tf.ConfigProto(device_count={"CPU":4},inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)) as sess:
        # TODO: with every epoches, print training accuracy and validation accuracy
        sess.run(tf.global_variables_initializer())
        # restore trained parameters
        check_restore_parameters(sess, saver)
        print('start training...')
        train_model(sess, model, train_path, valid_path, feature_length)