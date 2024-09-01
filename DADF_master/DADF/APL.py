import os
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
import time
import argparse
import csv
import re
import scipy.io as sio
from keras.api._v2.keras.models import Model, load_model
from keras.api._v2.keras.layers import Input, Layer, Conv1D, GlobalAveragePooling1D, Concatenate, Lambda
from keras.api._v2.keras.losses import Loss
from keras.api._v2.keras.callbacks import ModelCheckpoint, CSVLogger

from models import sem_enc_model, chan_enc_model, Chan_Model, chan_dec_model, sem_dec_model

def preprocess_audio(audio_path, target_length=16384, sr=8000):
    audio, _ = librosa.load(audio_path, sr=sr)
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    return audio

def generate_data(directory_ori, directory_adv, output_dir):
    filenames_ori = {f: os.path.join(directory_ori, f) for f in os.listdir(directory_ori) if f.endswith('.wav')}
    filenames_adv = {f: os.path.join(directory_adv, f) for f in os.listdir(directory_adv) if f.endswith('.wav')}
    original_audio = []
    adversarial_audio = []
    pair_labels = []

    for filename, path_ori in filenames_ori.items():
        if filename in filenames_adv:
            path_adv = filenames_adv[filename]
            ori_audio = preprocess_audio(path_ori)
            adv_audio = preprocess_audio(path_adv)
            original_audio.append(ori_audio)
            adversarial_audio.append(adv_audio)
            pair_labels.append(1) 

    original_audio = np.array(original_audio)
    adversarial_audio = np.array(adversarial_audio)
    np.save(os.path.join(output_dir, 'original_audio.npy'), original_audio)
    np.save(os.path.join(output_dir, 'adversarial_audio.npy'), adversarial_audio)
    return original_audio, adversarial_audio, np.array(pair_labels)

class TextCNNEncoder(Layer):
    def __init__(self, num_filters, filter_sizes, **kwargs):
        super(TextCNNEncoder, self).__init__(**kwargs)
        self.num_filters = num_filters 
        self.filter_sizes = filter_sizes

        self.conv_blocks = [
            [Conv1D(filters=num_filters, kernel_size=f, activation='relu', padding='same'),
             Conv1D(filters=num_filters, kernel_size=f, activation='relu', padding='same')]
            for f in filter_sizes
        ]
        self.concat = Concatenate()
        self.channel_avg = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))

    def call(self, inputs):
        outputs = []
        for block in self.conv_blocks:
            x = inputs
            for conv in block:
                x = conv(x) 
            outputs.append(x) 

        combined = self.concat(outputs) 
        final_output = self.channel_avg(combined)
        return final_output
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "filter_sizes": self.filter_sizes,
        })
        return config

class SimpleModel(Layer):
    def __init__(self, num_filters, filter_sizes, **kwargs):
        super(SimpleModel, self).__init__(**kwargs)
        self.num_filters = num_filters 
        self.filter_sizes = filter_sizes
        self.textcnn1 = TextCNNEncoder(num_filters, filter_sizes)
        self.textcnn2 = TextCNNEncoder(num_filters, filter_sizes)

    def call(self, inputs):
        x1, x2 = inputs
        x1 = self.textcnn1(x1)
        x2 = self.textcnn2(x2)
        return x2
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "filter_sizes": self.filter_sizes,
        })
        return config

class APL_Loss(Loss):
    def __init__(self, margin=1.0, **kwargs):
        super(APL_Loss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, pair_labels, y_pred):

        pair_labels = tf.cast(pair_labels, tf.float32)
        pair_labels = tf.reshape(pair_labels, [-1, 1]) 

        ori, adv = tf.split(y_pred, num_or_size_splits=2, axis=1)
        distances = tf.reduce_mean(tf.square(ori - adv), axis=1)
        distances = tf.expand_dims(distances, 1)  

        positive_loss = distances
        negative_loss = tf.maximum(self.margin - distances, 0)

        loss = pair_labels * positive_loss + (1 - pair_labels) * negative_loss
        loss = tf.reduce_mean(loss)  
        return loss
    def get_config(self): 
        config = super(APL_Loss, self).get_config() 
        config.update({"margin": self.margin}) 
        return config

def find_last_checkpoint(save_model_dir):
    checkpoint_files = [f for f in os.listdir(save_model_dir) if 'ckpt-' in f and f.endswith('.h5')]
    if not checkpoint_files:
        return None, 0

    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
    epoch_num = int(latest_checkpoint.split('-')[1].split('.')[0])
    return os.path.join(save_model_dir, latest_checkpoint), epoch_num

def APL(directory_ori, directory_adv):
    #All the code will be released after acceptance.


def train_combined():
    directory_ori = '/audioattack/ori' 
    directory_adv = '/audioattack/train' 

    print("Starting contrastive learning...")
    processed_output = APL(directory_ori, directory_adv)

if __name__ == '__main__':

    train_combined()


