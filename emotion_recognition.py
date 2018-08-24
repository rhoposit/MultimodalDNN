#!/usr/bin/env python
################################################################################
#       The Edinburgh G25 Multimodal DNN Emotion Recognition System
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
from mmdata import MOSEI
import argparse
from collections import defaultdict
from mmdata.dataset import Dataset
from utils.parser_utils import KerasParserClass
from utils.storage import build_experiment_folder, save_statistics

#switch between val_loss+min or val_acc+max
val_method = "val_loss"
val_mode = "min"



# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
import segeval
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Conv2D, Flatten,BatchNormalization
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
        new_data = data[-max_len:]
        return new_data

    

def custom_split(train, valid):
    valid = list(valid)
    train = list(train)
    train_ids = []
    valid_ids = []
    test_ids = []
    total = len(valid)
    half = total / 2
    valid_ids = valid[:half]
    test_ids = valid[half+1:]
    # 5 % of training into test data
    five_p = int(len(train) * 0.05)
    train_ids = train[:-five_p]
    test_ids = test_ids + train[-five_p:]
    # 10% of leftover training into valid data
    ten_p = int(len(train_ids) * 0.1)
    train_ids = train_ids[:-ten_p]
    valid_ids = valid_ids + train_ids[-ten_p:]
    return train_ids, valid_ids, test_ids


def get_class_MAE(truth, preds):
    ref = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    class_MAE = []
    truth = np.array(truth)
    preds = np.array(preds)
    for i in range(len(truth[0])):
        T = truth[:, i]
        P = preds[:,i]
        class_MAE.append(sklearn.metrics.mean_absolute_error(T, P))
    outstring = ""
    for i in range(len(class_MAE)):
        o = ref[i]+"="+str(class_MAE[i]) + "\n"
        outstring += o
    return outstring




def run_experiment(max_len, dropout_rate, n_layers):

    global dataset,train_ids, valid_ids, test_ids, mode, val_method, val_mode

    nodes = 100
    epochs = 200
    outfile = "final_sweep/blstm_"+mode+"_"+str(n_layers)+"_"+str(max_len)+"_"+str(dropout_rate)
    experiment_prefix = "blstm_early_fusion"
    batch_size = 64
    logs_path = "regression_logs/"
    experiment_name = "{}_n_{}_dr_{}_nl_{}_ml_{}".format(experiment_prefix,nodes,dropout_rate, n_layers, max_len)

        
    # sort through all the video ID, segment ID pairs
    train_set_ids = []
    for vid in train_ids:
        for sid in dataset['embeddings'][vid].keys():
            if mode == "all" or mode == "AV":
                if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid] and pad(dataset['covarep'][vid][sid], max_len).shape[1] == 74 and sid != 0:
                    train_set_ids.append((vid, sid))
            if mode == "AT" or mode == "A":
                if dataset['embeddings'][vid][sid] and dataset['covarep'][vid][sid] and pad(dataset['covarep'][vid][sid], max_len).shape[1] == 74 and sid != 0:
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
                if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid] and pad(dataset['covarep'][vid][sid], max_len).shape[1] == 74 and sid != 0:
                    valid_set_ids.append((vid, sid))
            if mode == "AT" or mode == "A":
                if dataset['embeddings'][vid][sid] and dataset['covarep'][vid][sid] and pad(dataset['covarep'][vid][sid], max_len).shape[1] == 74 and sid != 0:
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
                    if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid] and dataset['covarep'][vid][sid] and pad(dataset['covarep'][vid][sid], max_len).shape[1] == 74 and sid!= 0:
                        test_set_ids.append((vid, sid))
                if mode == "AT" or mode == "A":
                    if dataset['embeddings'][vid][sid] and dataset['covarep'][vid][sid] and pad(dataset['covarep'][vid][sid], max_len).shape[1] == 74 and sid != 0:
                        test_set_ids.append((vid, sid))
                if mode == "VT" or mode == "V":
                    if dataset['embeddings'][vid][sid] and dataset['facet'][vid][sid]:
                        test_set_ids.append((vid, sid))
                if mode == "T":
                    if dataset['embeddings'][vid][sid]:
                        test_set_ids.append((vid, sid))


    # partition the training, valid and test set. all sequences will be padded/truncated to 15 steps
    # data will have shape (dataset_size, max_len, feature_dim)
    if mode == "all" or mode == "AV" or mode == "AT" or mode == "A":
        train_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['covarep'][vid][sid]], axis=0)
        valid_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['covarep'][vid][sid]], axis=0)
        test_set_audio = np.stack([pad(dataset['covarep'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['covarep'][vid][sid]], axis=0)
    if mode == "all" or mode == "VT" or mode == "AV" or mode == "V":
        train_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['facet'][vid][sid]], axis=0)
        valid_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['facet'][vid][sid]], axis=0)
        test_set_visual = np.stack([pad(dataset['facet'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['facet'][vid][sid]], axis=0)

    if mode == "all" or mode == "VT" or mode == "AT" or mode == "T":        
        train_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in train_set_ids if dataset['embeddings'][vid][sid]], axis=0)
        valid_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in valid_set_ids if dataset['embeddings'][vid][sid]], axis=0)
        test_set_text = np.stack([pad(dataset['embeddings'][vid][sid], max_len) for (vid, sid) in test_set_ids if dataset['embeddings'][vid][sid]], axis=0)

    y_train = np.array([emotions[vid][sid] for (vid, sid) in train_set_ids])
    y_valid = np.array([emotions[vid][sid] for (vid, sid) in valid_set_ids])
    y_test = np.array([emotions[vid][sid] for (vid, sid) in test_set_ids])

    # normalize covarep and facet features, remove possible NaN values
    if mode == "all" or mode == "AV" or mode == "VT" or mode == "V":
        visual_max = np.max(np.max(np.abs(train_set_visual), axis=0), axis=0)
        visual_max[visual_max==0] = 1 # if the maximum is 0 we don't normalize
        train_set_visual = train_set_visual / visual_max
        valid_set_visual = valid_set_visual / visual_max
        test_set_visual = test_set_visual / visual_max
        train_set_visual[train_set_visual != train_set_visual] = 0
        valid_set_visual[valid_set_visual != valid_set_visual] = 0
        test_set_visual[test_set_visual != test_set_visual] = 0

    if mode == "all" or mode == "AT" or mode == "AV" or mode == "A":
        audio_max = np.max(np.max(np.abs(train_set_audio), axis=0), axis=0)
        train_set_audio = train_set_audio / audio_max
        valid_set_audio = valid_set_audio / audio_max
        test_set_audio = test_set_audio / audio_max
        train_set_audio[train_set_audio != train_set_audio] = 0
        valid_set_audio[valid_set_audio != valid_set_audio] = 0
        test_set_audio[test_set_audio != test_set_audio] = 0

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





    emote_final = 'linear'
    model = Sequential()

    if n_layers == 1:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=emote_final))
            
    if n_layers == 2:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(Bidirectional(LSTM(64,return_sequences=True,input_shape=(max_len, x_train.shape[2]))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=emote_final))
                    
    if n_layers == 3:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(Bidirectional(LSTM(64,return_sequences=True,input_shape=(max_len, x_train.shape[2]))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64,return_sequences=True,input_shape=(max_len, x_train.shape[2]))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=emote_final))

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

    out = open(outfile, "wb")
    out.write("---------ORIGINAL predictions (not scaled or bounded)---------")
    preds = model.predict(x_test)
    mae = sklearn.metrics.mean_absolute_error(y_test, preds)
    class_mae = get_class_MAE(y_test,preds)
    out.write("Test Sklearn MAE: "+str(mae)+"\n")
    out.write("Per-class MAE: "+str(class_mae)+"\n")
    out.write("dropout_rate="+str(dropout_rate)+"\n")
    out.write("n_layers="+str(n_layers)+"\n")
    out.write("max_len="+str(max_len)+"\n")
    out.write("nodes="+str(nodes)+"\n")
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
# specify the multimodal combination (all, AV, VT, AT, A, V, T)
mode = sys.argv[1]


# Download the data if not present
mosei = MOSEI()
embeddings = mosei.embeddings()
if mode == "all" or mode == "AV" or mode == "VT" or mode == "V":
    facet = mosei.facet()
if mode == "all" or mode == "AT" or mode == "AV" or mode == "A":
    covarep = mosei.covarep()
sentiments = mosei.sentiments() 
emotions = mosei.emotions()
train_ids = mosei.train()
valid_ids = mosei.valid()
train_ids, valid_ids, test_ids = custom_split(train_ids, valid_ids)

    
# Merge different features and do word level feature alignment (align according to timestamps of embeddings)
if mode == "all" or mode == "AV":
    bimodal = Dataset.merge(embeddings, facet)
    trimodal = Dataset.merge(bimodal, covarep)
    dataset = trimodal.align('embeddings')
if mode == "AT" or mode == "A":
    bimodal = Dataset.merge(embeddings, covarep)
    dataset = bimodal.align('embeddings')
if mode == "VT" or mode == "V":
    bimodal = Dataset.merge(embeddings, facet)
    dataset = bimodal.align('embeddings')
if mode == "T":
    dataset = embeddings


# SWEEP values    
LENS = [15,20,25,30]
DROP = [0.1,0.2]
LAYER = [1,2,3]

#run_experiment(30, 0.2, 2)

# Run sweep in parallel
Parallel(n_jobs=num_cores)(delayed(run_experiment)(max_len=i, dropout_rate=j, n_layers=k) for i in LENS for j in DROP for k in LAYER) 
            
 
