# Noemi's network: dense ReLU + dense ReLU + softmax
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import logging
import datetime
import os
import numpy as np
import tensorflow as tf
from random import shuffle

# hyperparameters can be changed after cross validation
# maybe gradient clipping can be added
# 

class Encoder(object):
    def __init__(self, INPUT_DIM, STATE_SIZE = 100):
        self.INPUT_DIM = INPUT_DIM
        self.STATE_SIZE = STATE_SIZE
        self.W1 = tf.get_variable("W1", dtype = tf.float64, shape = (self.INPUT_DIM, self.STATE_SIZE), initializer = tf.contrib.layers.xavier_initializer()) 
        # self.W1 = tf.get_variable("W1", dtype = tf.float64, shape = (self.INPUT_DIM, self.STATE_SIZE)) 
        self.b1 = tf.get_variable("b1", dtype = tf.float64, shape = (self.STATE_SIZE), initializer = tf.zeros_initializer())
        # self.b1 = tf.get_variable("b1", dtype = tf.float64, shape = (self.STATE_SIZE), initializer=tf.uniform_unit_scaling_initializer(1.0))

    def encode(self, x):
        h = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        return h
class Classifier(object):
    def __init__(self, STATE_SIZE = 100):
        self.STATE_SIZE = STATE_SIZE
        self.W3 = tf.get_variable("W3", dtype = tf.float64, shape = (self.STATE_SIZE, 1), initializer = tf.contrib.layers.xavier_initializer())
        # self.W3 = tf.get_variable("W3", dtype = tf.float64, shape = (self.STATE_SIZE, 1))
        self.b3 = tf.get_variable("b3", dtype = tf.float64, shape = (1), initializer = tf.zeros_initializer())
        # self.b3 = tf.get_variable("b3", dtype = tf.float64, shape = (1), initializer=tf.uniform_unit_scaling_initializer(1.0))
    def classify(self, input):
        return tf.matmul(input, self.W3) + self.b3
class NeuralNet(object):
    def __init__(self, encoder1, encoder2, classifier, LEARNING_RATE = 10**-3, REG_PARAM = 0, DROP_PROB = 0.25, FEATURE_DIM = 50, MAX_GRADIENT = 10):
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.classifier = classifier
        self.LEARNING_RATE = LEARNING_RATE
        self.REG_PARAM = REG_PARAM
        self.KEEP_PROB = 1.0-DROP_PROB
        self.FEATURE_DIM = FEATURE_DIM
        self.MAX_GRADIENT = MAX_GRADIENT
        self.x = tf.placeholder(tf.float64, shape = (None, self.FEATURE_DIM))
        self.y = tf.placeholder(tf.float64, shape = (None,1))
        self.sel = tf.placeholder(tf.float64, shape = (None,))
        self.optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        # self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        
        params = tf.trainable_variables()
        self.globalnorm = 0
        self.paramnorm = 0
        for param in params:
            shp = param.get_shape()
            if len(shp) >= 2:
                self.paramnorm += tf.nn.l2_loss(param)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours = 2, max_to_keep = 0)
        self.backprop()
        grads, varss = zip(*self.optimizer.compute_gradients(self.loss))
        # self.globalnorm = tf.global_norm(grads)
        # if self.globalnorm > self.MAX_GRADIENT:
        grads_clipped, _ = tf.clip_by_global_norm(grads, self.MAX_GRADIENT)
        self.globalnorm = tf.global_norm(grads_clipped)
        # print(grads)
        # print(grads_clipped)
        # print(varss)
        self.updates = self.optimizer.apply_gradients(zip(grads_clipped, varss))

    def forwardprop(self):
        x_drop = tf.nn.dropout(self.x, self.KEEP_PROB)
        h1 = self.encoder1.encode(x_drop)
        h1_drop = tf.nn.dropout(h1, self.KEEP_PROB)
        h2 = self.encoder2.encode(h1_drop)
        h2_drop = tf.nn.dropout(h2, self.KEEP_PROB)
        self.o = self.classifier.classify(h2_drop)
    def compute_loss(self):
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y, logits = self.o)
        self.loss = tf.matmul(tf.expand_dims(self.sel, axis = 0), self.loss)
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y, logits = self.o))
        # print(self.loss)
        # if regularize, also compute self.regloss here
    def backprop(self):
        self.forwardprop()
        self.compute_loss()
        
    def optimize(self, sess, x, y, sel):
        input_feed = {}
        input_feed[self.x] = x
        input_feed[self.y] = y
        input_feed[self.sel] = sel
        output_list = [self.updates, self.loss, self.paramnorm, self.globalnorm]
        outputs = sess.run(output_list, input_feed)
        return outputs
    def test(self, sess, x, y, sel):
        input_feed = {}
        input_feed[self.x] = x
        input_feed[self.y] = y
        input_feed[self.sel] = sel
        output_list = [self.o, self.loss]
        outputs = sess.run(output_list, input_feed)
        return outputs
    def accuracy(self, sess, x, y, sel):
        y_expand = np.expand_dims(y, axis=1)
        o, loss = self.test(sess, x, y_expand, sel)
        errors = 0
        for i in range(len(y)):
            label = y[i]
            if o[i] > 0:
                pred = 1
            else:
                pred = 0
            if pred != int(label):
                errors += 1
        return errors, len(y), loss
    def readPreprocessedData(self, train_folder, fileName, batchSize=100):

        # for fileName in os.listdir(dirName):
        
        with open(os.path.join(train_folder, fileName), 'r') as f:
            data = []
            for line in f:
                if len(line) == 0:
                    continue
                if line[0] == 'F':
                    continue
                parts = line.strip().split(',')
                data.append((parts[:-1], parts[-1]))
                if len(data) == batchSize:
                    yield data
                    data = []

        yield data

    def train(self, sess, train_folder, devName, numEpochs, save_directory, dev_index):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))
        bestDevAcc = 0.0
        bestEpoch = None
        bestDevLoss = 100.0
        bestLossEpoch = None
        devAccBestLoss = 0.0
        devLossBestAcc = 100.0
        keepProb = self.KEEP_PROB
        for e in range(numEpochs):
            self.KEEP_PROB = keepProb
            lossTrain = 0.0
            # if e == 10:
                # self.LEARNING_RATE = self.LEARNING_RATE / 4
            lst = os.listdir(train_folder)
            shuffle(lst)
            for filename in lst:
                # if filename == "__add__": continue
                if filename == ".DS_Store": continue
                if filename == devName:
                    continue
                else:
                    batch = 1
                    for data in self.readPreprocessedData(train_folder, filename):
                        cells, labels = zip(*data)
                        sel = []
                        for j in range(len(labels)):
                            if int(labels[j]) == 1:
                                sel.append(1)
                            else:
                                u = np.random.random()
                                if u>0.7:
                                    sel.append(0)
                                else: sel.append(1)
                        labels = np.expand_dims(labels, axis=1)
                        _, currentLoss, parNorm, gradNorm = self.optimize(sess, cells, labels, sel)
                        # print(len(labels))
                        lossTrain += currentLoss * len(labels)
                        if batch % 50 == 0:
                            logging.info('File name: %s Epoch: %d Batch: %d Current loss: %f Parameter norm: %f Gradient norm: %f' %(filename, e, batch, currentLoss, parNorm, gradNorm))
                        # logging.info(currentLoss)
                        batch += 1
            trainAcc = 0.0
            devAcc = 0.0
            trainLoss = 0.0
            devLoss = 0.0
            trainTotal = 0
            devTotal = 0
            self.KEEP_PROB = 1.0
            for filename in os.listdir(train_folder):
                if filename != devName:
                    for data in self.readPreprocessedData(train_folder, filename):
                        if len(data) == 0: continue
                        cells,labels = zip(*data)
                        sel = [1 for label in labels]
                        errsTrain, totalTrain, trainL = self.accuracy(sess, cells, labels, sel)
                        # print(totalTrain)
                        trainTotal += totalTrain
                        trainAcc += errsTrain
                        trainLoss += trainL * totalTrain
                else:
                # batch = 1
                    for data in self.readPreprocessedData(train_folder, filename):
                        if len(data) == 0: continue
                        cells, labels = zip(*data)
                        sel = [1 for label in labels]
                    # labels = np.expand_dims(labels, axis=1)
                        errs, total, devL = self.accuracy(sess, cells, labels, sel)
                        devTotal += total
                        devAcc += errs
                        devLoss += devL * total
                    # print(e, batch, devLoss, errs * 1.0 / total)
                    # batch+= 1
                    # print(devAcc)
                    # print(devTotal)
            trainAcc = 1.0 - (trainAcc * 1.0) / trainTotal
            devAcc = 1.0 - (devAcc * 1.0) / devTotal
            lossTrain = lossTrain / trainTotal
            trainLoss = trainLoss / trainTotal
            devLoss = devLoss / devTotal
            self.saver.save(sess, save_directory + "/profilernn1-dev-" + str(dev_index) + "-epoch", global_step=e)
            if devAcc > bestDevAcc:
                # self.saver.save(sess, save_directory + "/profilernn1-dev-" + str(dev_index) + "-epoch", global_step=e)
                bestDevAcc = devAcc
                bestEpoch = e
                devLossBestAcc = devLoss
            if devLoss < bestDevLoss:
                # self.saver.save(sess, save_directory + "/profilernn1-dev-" + str(dev_index) + "-epoch", global_step=e)
                bestDevLoss = devLoss
                bestLossEpoch = e
                devAccBestLoss = devAcc
            logging.info('Train loss: %f Dev loss: %f Train acc: %f Dev acc: %f Epoch: %d Loss train: %f' % (trainLoss, devLoss, trainAcc, devAcc, e, lossTrain))
        logging.info('Training completed. Best devAcc %f is obtained in epoch %d with dev loss %f. Best dev loss %f is obtained in epoch %d with accuracy %f.' % (bestDevAcc, bestEpoch, devLossBestAcc, bestDevLoss, bestLossEpoch, devAccBestLoss))
# raise NotImplementedError
    # def boostingTest(self, sess, model_folder, test_folder, filename):
        
                
        





