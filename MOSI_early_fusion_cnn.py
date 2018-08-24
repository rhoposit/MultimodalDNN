#!/usr/bin/env python
################################################################################
#          The Edinburgh G25 Multimodal DNN Sentiment Analysis System
#                https://github.com/rhoposit/emotionChallenge
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2017-2018
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################

import sys
import numpy as np
from collections import defaultdict
from operator import itemgetter
from mmdata import MOSEI, MOSI
import argparse
from collections import defaultdict
from mmdata.dataset import Dataset
from utils.parser_utils import KerasParserClass
from utils.storage import build_experiment_folder, save_statistics

val_method = "val_loss"
val_mode = "min"



# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
import sklearn
import scipy
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
import segeval
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Conv2D, Flatten,BatchNormalization, Merge, Input, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger

def pad(data, max_len):
    """A funtion for padding/truncating sequence data to a given lenght"""
    # recall that data at each time step is a tuple (start_time, end_time, feature_vector), we only take the vector
    data = np.array([feature[2] for feature in data])
    n_rows = data.shape[0]
    dim = data.shape[1]
    if max_len >= n_rows:
            diff = max_len - n_rows
            padding = np.zeros((diff, dim))
            padded = np.concatenate((padding, data))
            return padded
    else:
        idx = np.random.choice(np.arange(n_rows), max_len, replace=False)
        return data[idx]
#        return data[-max_len:]


def get_idx(val):
    if val <= -1.8:
        return 0
    elif val <= -0.6:
        return 1
    elif val <= 0.6:
        return 2
    elif val <= 1.8:
        return 3
    elif val <= 3.0:
        return 4



def convert_S5_hot(orig):
    converted = []
    for i in orig:
        new = np.zeros(5)
        idx = get_idx(i)
        new[idx] = 1
        converted.append(new)
        print idx
        print "cold: ", i
        print "hot: ", new
    return np.array(converted)


def convert_pred_hot(orig):
    converted = []
    for i in orig:
        new = np.zeros(5)
        idx = np.argmax(np.array(i))
        new[idx] = 1
        converted.append(new)
    return np.array(converted)



def run_experiment(max_len, dropout_rate, n_layers):

    global dataset,train_ids, valid_ids, test_ids, mode, sent_emo, val_method, val_mode, use_PCA

    # For PCA if set to True
    visual_components = 25
    audio_components = 20
    text_components = 110

    nodes = 100
    epochs = 200
    outfile = "MOSI_sweep/cnn_"+mode+"_"+str(sent_emo)+"_"+str(n_layers)+"_"+str(max_len)+"_"+str(dropout_rate)
    experiment_prefix = "cnn"
    batch_size = 64
    logs_path = "regression_logs/"
    experiment_name = "{}_n_{}_dr_{}_nl_{}_ml_{}".format(experiment_prefix,nodes,dropout_rate, n_layers, max_len)

        
    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
        for sid in dataset['embeddings'][vid].keys():
            if mode == "all" or mode == "AV":
                if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
                    train_set_ids.append((vid, sid))
            if mode == "AT" or mode == "A":
                if dataset['embeddings'][vid][sid] and dataset['covarep'][vid][sid]:
                    train_set_ids.append((vid, sid))
            if mode == "VT" or mode == "V":
                if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid]:
                    train_set_ids.append((vid, sid))
            if mode == "T":
                if dataset['embeddings'][vid][sid]:
                    train_set_ids.append((vid, sid))

    valid_set_ids = []
    for vid in valid_ids:
        for sid in dataset['embeddings'][vid].keys():
            if mode == "all" or mode == "AV":
                if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
                    valid_set_ids.append((vid, sid))
            if mode == "AT" or mode == "A":
                if dataset['embeddings'][vid][sid] and dataset['covarep'][vid][sid]:
                    valid_set_ids.append((vid, sid))
            if mode == "VT" or mode == "V":
                if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid]:
                    valid_set_ids.append((vid, sid))
            if mode == "T":
                if dataset['embeddings'][vid][sid]:
                    valid_set_ids.append((vid, sid))

    test_set_ids = []
    for vid in test_ids:
        if vid in dataset['embeddings']:
            for sid in dataset['embeddings'][vid].keys():
                if mode == "all" or mode == "AV":
                    if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid]:
                        test_set_ids.append((vid, sid))
                if mode == "AT" or mode == "A":
                    if dataset['embeddings'][vid][sid] and dataset['covarep'][vid][sid]:
                        test_set_ids.append((vid, sid))
                if mode == "VT" or mode == "V":
                    if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid]:
                        test_set_ids.append((vid, sid))
                if mode == "T":
                    if dataset['embeddings'][vid][sid]:
                        test_set_ids.append((vid, sid))


    # partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
    # data will have shape (dataset_size, max_len, feature_dim)
    if mode == "all" or mode == "AV" or mode == "AT":
        train_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['covarep'][vid][sid]], axis=0)
        valid_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['covarep'][vid][sid]], axis=0)
        test_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['covarep'][vid][sid]], axis=0)
    if mode == "all" or mode == "VT" or mode == "AV":
        train_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['facet'][vid][sid]], axis=0)
        valid_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['facet'][vid][sid]], axis=0)
        test_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['facet'][vid][sid]], axis=0)

    if mode == "all" or mode == "VT" or mode == "AT":        
        train_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['embeddings'][vid][sid]], axis=0)
        valid_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['embeddings'][vid][sid]], axis=0)
        test_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['embeddings'][vid][sid]], axis=0)

    if task == "SB":
        # binarize the sentiment scores for binary classification task
        y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids]) > 0
        y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids]) > 0
        y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids]) > 0

    if task == "SR":
        y_train = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
        y_valid = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
        y_test = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])

    if task == "S5":
        y_train1 = np.array([sentiments[vid][sid] for (vid, sid) in train_set_ids])
        y_valid1 = np.array([sentiments[vid][sid] for (vid, sid) in valid_set_ids])
        y_test1 = np.array([sentiments[vid][sid] for (vid, sid) in test_set_ids])
        y_train = convert_S5_hot(y_train1)
        y_valid = convert_S5_hot(y_valid1)
        y_test = convert_S5_hot(y_test1)

    # normalize covarep and facet features, remove possible NaN values
    if mode == "all" or mode == "AV" or mode == "VT":
        visual_max = np.max(np.max(np.abs(train_set_visual), axis=0), axis=0)
        visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize
        train_set_visual = train_set_visual / visual_max
        valid_set_visual = valid_set_visual / visual_max
        test_set_visual = test_set_visual / visual_max
        train_set_visual[train_set_visual != train_set_visual] = 0
        valid_set_visual[valid_set_visual != valid_set_visual] = 0
        test_set_visual[test_set_visual != test_set_visual] = 0

    if mode == "all" or mode == "AT" or mode == "AV":
        audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
        train_set_audio = train_set_audio / audio_max
        valid_set_audio = valid_set_audio / audio_max
        test_set_audio = test_set_audio / audio_max
        train_set_audio[train_set_audio != train_set_audio] = 0
        valid_set_audio[valid_set_audio != valid_set_audio] = 0
        test_set_audio[test_set_audio != test_set_audio] = 0

    if use_PCA == True:
        if mode == "all" or mode == "AV" or mode == "VT":
            nsamples1, nx1, ny1 = train_set_visual.shape
            train_set_visual = train_set_visual.reshape(nsamples1*nx1, ny1)
            nsamples2, nx2, ny2 = valid_set_visual.shape
            valid_set_visual = valid_set_visual.reshape(nsamples2*nx2, ny2)
            nsamples3, nx3, ny3 = test_set_visual.shape
            test_set_visual = test_set_visual.reshape(nsamples3*nx3, ny3)
            pca = decomposition.PCA(n_components=visual_components)
            train_set_visual_pca = pca.fit_transform(train_set_visual)
            valid_set_visual_pca = pca.transform(valid_set_visual)
            test_set_visual_pca = pca.transform(test_set_visual)
            train_set_visual = train_set_visual_pca.reshape(nsamples1,nx1,visual_components)
            valid_set_visual = valid_set_visual_pca.reshape(nsamples2,nx2,visual_components)
            test_set_visual = test_set_visual_pca.reshape(nsamples3,nx3,visual_components)
    
        if mode == "all" or mode == "AT" or mode == "AV":
            nsamples1, nx1, ny1 = train_set_audio.shape
            train_set_audio = train_set_audio.reshape(nsamples1*nx1, ny1)
            nsamples2, nx2, ny2 = valid_set_audio.shape
            valid_set_audio = valid_set_audio.reshape(nsamples2*nx2, ny2)
            nsamples3, nx3, ny3 = test_set_audio.shape
            test_set_audio = test_set_audio.reshape(nsamples3*nx3, ny3)
            pca = decomposition.PCA(n_components=audio_components)
            train_set_audio_pca = pca.fit_transform(train_set_audio)
            valid_set_audio_pca = pca.transform(valid_set_audio)
            test_set_audio_pca = pca.transform(test_set_audio)
            train_set_audio = train_set_audio_pca.reshape(nsamples1, nx1, audio_components)
            valid_set_audio = valid_set_audio_pca.reshape(nsamples2, nx2, audio_components)
            test_set_audio = test_set_audio_pca.reshape(nsamples3, nx3, audio_components)

        if mode == "all" or mode == "AT" or mode == "VT":    
            nsamples1, nx1, ny1 = train_set_text.shape
            train_set_text = train_set_text.reshape(nsamples1*nx1, ny1)
            nsamples2, nx2, ny2 = valid_set_text.shape
            valid_set_text = valid_set_text.reshape(nsamples2*nx2, ny2)
            nsamples3, nx3, ny3 = test_set_text.shape
            test_set_text = test_set_text.reshape(nsamples3*nx3, ny3)
            pca = decomposition.PCA(n_components=text_components)
            train_set_text_pca = pca.fit_transform(train_set_text)
            valid_set_text_pca = pca.transform(valid_set_text)
            test_set_text_pca = pca.transform(test_set_text)
            train_set_text = train_set_text_pca.reshape(nsamples1, nx1, text_components)
            valid_set_text = valid_set_text_pca.reshape(nsamples2, nx2, text_components)
            test_set_text = test_set_text_pca.reshape(nsamples3, nx3, text_components)
            
    if mode == "all":
        x_train = np.concatenate((train_set_visual, train_set_audio, train_set_text), axis=2)
        x_valid = np.concatenate((valid_set_visual, valid_set_audio, valid_set_text), axis=2)
        x_test = np.concatenate((test_set_visual, test_set_audio, test_set_text), axis=2)
    if mode == "AV":
        x_train = np.concatenate((train_set_visual, train_set_audio), axis=2)
        x_valid = np.concatenate((valid_set_visual, valid_set_audio), axis=2)
        x_test = np.concatenate((test_set_visual, test_set_audio), axis=2)
    if mode == "AT":
        x_train = np.concatenate((train_set_audio, train_set_text), axis=2)
        x_valid = np.concatenate((valid_set_audio, valid_set_text), axis=2)
        x_test = np.concatenate((test_set_audio, test_set_text), axis=2)
    if mode == "VT":
        x_train = np.concatenate((train_set_visual, train_set_text), axis=2)
        x_valid = np.concatenate((valid_set_visual, valid_set_text), axis=2)
        x_test = np.concatenate((test_set_visual, test_set_text), axis=2)
    if mode == "V":
        x_train = train_set_visual
        x_valid = valid_set_visual
        x_test = test_set_visual
    if mode == "A":
        x_train = train_set_audio
        x_valid = valid_set_audio
        x_test = test_set_audio
    if mode == "T":
        x_train = train_set_text
        x_valid = valid_set_text
        x_test = test_set_text


    k = 3
    m = 2
    if task == "SB":
        val_method = "val_acc"
        val_mode = "max"
        emote_final = 'sigmoid'
    if task == "SR":
        val_method = "val_loss"
        val_mode = "min"
        emote_final = 'linear'        
    if task == "S5":
        val_method = "val_acc"
        val_mode = "max"
        emote_final = 'softmax'
    model = Sequential()

    emote_final = 'linear'
    model = Sequential()

    if n_layers == 1:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(Conv1D(filters=128, kernel_size=k, input_shape = (max_len, x_train.shape[2]), activation='relu'))
        model.add(MaxPooling1D(m))
        model.add(Flatten())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nodes, activation='relu'))
            
    if n_layers == 2:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(Conv1D(filters=128, kernel_size=k, input_shape = (max_len, x_train.shape[2]), activation='relu'))
        model.add(MaxPooling1D(m))
        model.add(Conv1D(filters=128, kernel_size=k, input_shape = (max_len, x_train.shape[2]), activation='relu'))
        model.add(MaxPooling1D(m))
        model.add(Flatten())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nodes, activation='relu'))
            
    if n_layers == 3:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(Conv1D(filters=128, kernel_size=k, input_shape = (max_len, x_train.shape[2]), activation='relu'))
        model.add(MaxPooling1D(m))
        model.add(Conv1D(filters=128, kernel_size=k, input_shape = (max_len, x_train.shape[2]), activation='relu'))
        model.add(MaxPooling1D(m))
        model.add(Conv1D(filters=128, kernel_size=k, input_shape = (max_len, x_train.shape[2]), activation='relu'))
        model.add(MaxPooling1D(m))
        model.add(Flatten())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nodes, activation='relu'))


    if task == "SR":
        model.add(Dense(1, activation=emote_final))
    if task == "SB":
        model.add(Dense(1, activation=emote_final))
    if task == "S5":
        model.add(Dense(5, activation=emote_final))


    if task == "SB":
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    if task == "S5":
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    if task == "SR":
        model.compile('adam', loss='mean_absolute_error')

    early_stopping = EarlyStopping(monitor=val_method,
                                   min_delta=0,
                                   patience=10,
                                   verbose=1, mode=val_mode)
    callbacks_list = [early_stopping]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_valid, y_valid],
              callbacks=callbacks_list)

    preds = model.predict(x_test)
    out = open(outfile, "wb")

    print "testing output before eval metrics calcs.."
    print y_test[0]
    print preds[0]
    
    if task == "SR":
        preds = np.concatenate(preds)
        mae = sklearn.metrics.mean_absolute_error(y_test, preds)
        r = scipy.stats.pearsonr(y_test, preds)
        out.write("Test MAE: "+str(mae)+"\n")
        out.write("Test CORR: "+str(r)+"\n")
    if task == "S5":
        preds = convert_pred_hot(preds)
        acc = sklearn.metrics.accuracy_score(y_test, preds)
        out.write("Test ACC: "+str(acc)+"\n")
    if task == "SB":
        acc = np.mean((preds > 0.5) == y_test.reshape(-1, 1))
        preds = np.concatenate(preds)
        preds = preds > 0.5
        f1 = sklearn.metrics.f1_score(y_test, preds)
        out.write("Test ACC: "+str(acc)+"\n")
        out.write("Test F1: "+str(f1)+"\n")

    out.write("use_PCA="+str(use_PCA)+"\n")
    out.write("dropout_rate="+str(dropout_rate)+"\n")
    out.write("n_layers="+str(n_layers)+"\n")
    out.write("max_len="+str(max_len)+"\n")
    out.write("nodes="+str(nodes)+"\n")
    out.write("task="+str(task)+"\n")
    out.write("mode="+str(mode)+"\n")
    out.write("num_train="+str(len(train_set_ids))+"\n")
    out.write("num_valid="+str(len(valid_set_ids))+"\n")
    out.write("num_test="+str(len(test_set_ids))+"\n")
    out.close()




seed=1122017
np.random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
tf.set_random_seed(seed)
from joblib import Parallel, delayed
import multiprocessing


num_cores = multiprocessing.cpu_count()
mode = sys.argv[1]
task = sys.argv[2]


# Download the data if not present
mosei = MOSI()
embeddings = mosei.embeddings()
if mode == "all" or mode == "AV" or mode == "VT" or mode == "V":
    facet = mosei.facet()
if mode == "all" or mode == "AT" or mode == "AV" or mode == "A":
    covarep = mosei.covarep()
sentiments = mosei.sentiments() 
emotions = mosei.emotions()
train_ids = mosei.train()
valid_ids = mosei.valid()
test_ids = mosei.test()

    
# Merge different features and do word level feature alignment (align according to timestamps of embeddings)
if mode == "all" or mode == "AV":
    bimodal = Dataset.merge(embeddings, facet)
    trimodal = Dataset.merge(bimodal, covarep)
    dataset = trimodal.align('embeddings')
if mode == "AT":
    bimodal = Dataset.merge(embeddings, covarep)
    dataset = bimodal.align('embeddings')
if mode == "VT":
    bimodal = Dataset.merge(embeddings, facet)
    dataset = bimodal.align('embeddings')
if mode == "T":
    dataset = embeddings
if mode == "A":
    dataset = covarep
if mode == "V":
    dataset = facet


# SWEEP values    
LENS = [15, 20, 25, 30]
DROP = [0.1, 0.2]
LAYER = [1, 2, 3]


use_PCA = False

# Run sweep in parallel
Parallel(n_jobs=num_cores)(delayed(run_experiment)(max_len=i, dropout_rate=j, n_layers=k) for i in LENS for j in DROP for k in LAYER) 
            
 
