from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from neural_profiler1 import Encoder, Classifier, NeuralNet
from os.path import join as pjoin
from tensorflow.python.ops import variable_scope as vs
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
def readPreprocessedData(train_folder, fileName, batchSize=100):

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
def initialize_model(session, model, train_dir, no_reading, dev_index = None):
    if dev_index is not None:
        ckpt = tf.train.get_checkpoint_state(train_dir + "/" + train_dir + str(dev_index))
    else:
        ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if no_reading:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        return model        
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def main(_):
    # vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)
    STATE_SIZE = 100
    INPUT_DIM = 50
    LEARNING_RATE = 10**-5
    DROP_PROB = 0.2
    REG_PARAM = 0
    TRAIN_FOLDER = "data/train"
    TEST_FOLDER = "data/test"
    SAVE_FOLDER = "model"
    MAX_NORM = 5000.0
    NUM_EPOCHS = 30
    LOG_DIR = "log"
    LOAD_DIR = "model"
    # TRAIN_INDICES = [1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 15]
    TRAIN_INDICES = [4]
    acc_dict = {}
    # with vs.variable_scope("encoder1"):
        # encoder1 = Encoder(INPUT_DIM, STATE_SIZE)
    # with vs.variable_scope("encoder2"):
        # encoder2 = Encoder(STATE_SIZE, STATE_SIZE)
    # with vs.variable_scope("classifier"):
        # classifier = Classifier(STATE_SIZE)
    # classifier = Classifier(FLAGS)
    # nn = NeuralNet(encoder1, encoder2, classifier, LEARNING_RATE, REG_PARAM, DROP_PROB, INPUT_DIM, MAX_NORM)
    preds = {}
    for DEV_INDEX in TRAIN_INDICES:
        preds[DEV_INDEX] = {}
        with vs.variable_scope(str(DEV_INDEX) + "_encoder1"):
            encoder1 = Encoder(INPUT_DIM, STATE_SIZE)
        with vs.variable_scope(str(DEV_INDEX) + "_encoder2"):
            encoder2 = Encoder(STATE_SIZE, STATE_SIZE)
        with vs.variable_scope(str(DEV_INDEX) + "_classifier"):
            classifier = Classifier(STATE_SIZE)
                # classifier = Classifier(FLAGS)
        nn = NeuralNet(encoder1, encoder2, classifier, LEARNING_RATE, REG_PARAM, DROP_PROB, INPUT_DIM, MAX_NORM)
        file_handler = logging.FileHandler(pjoin(LOG_DIR, "log_" + str(DEV_INDEX) + ".txt"))
        logging.getLogger().addHandler(file_handler)
        with tf.Session() as sess:
            initialize_model(sess, nn, LOAD_DIR, False, DEV_INDEX)
            for filename in os.listdir(TEST_FOLDER):
                preds[DEV_INDEX][filename] = []
        # corrects = 0
                for data in readPreprocessedData(TEST_FOLDER, filename):
                    if len(data) == 0: continue
                    cells, labels = zip(*data)
                    sel = [1 for label in labels]
                    o, _ = nn.test(sess, cells, np.expand_dims(labels, axis = 1), sel)
                    for i in range(len(o)):
                        if o[i] > 0:
                            preds[DEV_INDEX][filename].append(1)
                        else:
                            preds[DEV_INDEX][filename].append(0)
    # print(preds)
    predOnes = {}
    for filename in os.listdir(TEST_FOLDER):
        predOnes[filename] = [0 for i in range(len(preds[DEV_INDEX][filename]))]
        for DEV_INDEX in TRAIN_INDICES:
            pred = preds[DEV_INDEX][filename]
            for i in range(len(pred)):
                if pred[i] == 1:
                    predOnes[filename][i] += 1
    finalPreds = {}
    for filename in os.listdir(TEST_FOLDER):
        finalPreds[filename] = []
        pred = predOnes[filename]
        for p in pred:
            if p > 0: # 5 if boosting
                finalPreds[filename].append(1)
            else:
                finalPreds[filename].append(0)
    for filename in os.listdir(TEST_FOLDER):
        pred = finalPreds[filename]
        i = 0
        corrects = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        total = 0
        for data in readPreprocessedData(TEST_FOLDER, filename):
            if len(data) == 0: continue
            cells, labels = zip(*data)
            leng = len(labels)
            for j in range(leng):
                if pred[i+j] == int(labels[j]):
                    corrects += 1
                    if pred[i+j] == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if pred[i+j] == 1:
                        fp += 1
                    else:
                        fn += 1
            i += leng
        acc_dict[filename] = ( (corrects * 1.0) / i , (1.0 * tp) / (tp+fp), (1.0 * tp) / (tp + fn))
    print(acc_dict)   
    prec = 0.0
    rec = 0.0
    acc = 0.0
    numKeys = 0
    for key in acc_dict:
        a, p, r = acc_dict[key]
        acc += a
        prec += p
        rec += r
        numKeys += 1
    print(acc / numKeys, prec / numKeys, rec / numKeys) 
    # for DEV_INDEX in TRAIN_INDICES:
        # for filename in os.listdir(TEST_FOLDER):
            # pred = preds[DEV_INDEX][filename]
            # i = 0
            # for data in readPreprocessedData(TEST_FOLDER, filename):
                # cells, labels = zip(*data)
                # leng = len(labels)
                # for j in range(leng):
                    # if pred[i+j] == labels[j]:
                        
    # for filename in os.listdir(TRAIN_FOLDER):
        
        # total = 0
        # corrects = 0
        # for data in readPreprocessedData(TRAIN_FOLDER, filename):
            # cells, labels = zip(*data)
            # predOnes = [0 for i in range(len(labels))]
            # for DEV_INDEX in TRAIN_INDICES:
                # with vs.variable_scope(str(DEV_INDEX) + "_encoder1", reuse = True):
                    # encoder1 = Encoder(INPUT_DIM, STATE_SIZE)
                # with vs.variable_scope(str(DEV_INDEX) + "_encoder2", reuse = True):
                    # encoder2 = Encoder(STATE_SIZE, STATE_SIZE)
                # with vs.variable_scope(str(DEV_INDEX) + "_classifier", reuse = True):
                    # classifier = Classifier(STATE_SIZE)
                # classifier = Classifier(FLAGS)
                # nn = NeuralNet(encoder1, encoder2, classifier, LEARNING_RATE, REG_PARAM, DROP_PROB, INPUT_DIM, MAX_NORM)

                # file_handler = logging.FileHandler(pjoin(LOG_DIR, "log_" + str(DEV_INDEX) + ".txt"))
                # logging.getLogger().addHandler(file_handler)
                # with tf.Session() as sess:
                    # initialize_model(sess, nn, LOAD_DIR, False, DEV_INDEX)
                    # o, _ = nn.test(sess, cells, np.expand_dims(labels, axis = 1))
                    # for i in range(len(o)):
                        # if o[i] > 0:
                            # predOnes[i] += 1
            # for i in range(len(predOnes)):
                # if predOnes[i] > 5:
                    # predOnes[i] = 1
                # else: predOnes[i] = 0
                # if predOnes[i] == int(labels[i]):
                    # corrects += 1
                # total += 1
        # acc_dict[filename] = (1.0 * corrects) / totals         
    # print(acc_dict)    
                
            
    # for DEV_INDEX in TRAIN_INDICES:
    # DEV_INDEX = 3
        # DEVNAME = "CleanData_" + str(DEV_INDEX) + ".txt"

        # with vs.variable_scope(str(DEV_INDEX) + "_encoder1"):
            # encoder1 = Encoder(INPUT_DIM, STATE_SIZE)
        # with vs.variable_scope(str(DEV_INDEX) + "_encoder2"):
            # encoder2 = Encoder(STATE_SIZE, STATE_SIZE)
        # with vs.variable_scope(str(DEV_INDEX) + "_classifier"):
            # classifier = Classifier(STATE_SIZE)
    # classifier = Classifier(FLAGS)
        # nn = NeuralNet(encoder1, encoder2, classifier, LEARNING_RATE, REG_PARAM, DROP_PROB, INPUT_DIM, MAX_NORM)

        # file_handler = logging.FileHandler(pjoin(LOG_DIR, "log_" + str(DEV_INDEX) + ".txt"))
        # logging.getLogger().addHandler(file_handler)
        
        # with tf.Session() as sess:
            # initialize_model(sess, nn, LOAD_DIR, True)
            # nn.train(sess, TRAIN_FOLDER, DEVNAME, NUM_EPOCHS, SAVE_FOLDER, DEV_INDEX)

if __name__ == "__main__":
    tf.app.run()
    
    # print(vars(FLAGS))
