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
import datetime
from keras.api._v2.keras.models import Model, load_model
from keras.api._v2.keras.layers import Input, Layer, Conv1D, GlobalAveragePooling1D, Concatenate, Lambda
from keras.api._v2.keras.losses import Loss
from keras.api._v2.keras.callbacks import ModelCheckpoint, CSVLogger
from keras.api._v2.keras.layers import Layer, Conv1D, Activation
from keras.api._v2.keras import backend as K
import tensorflow_probability as tfp
from models import sem_enc_model, chan_enc_model, Chan_Model, chan_dec_model, sem_dec_model

save_dir = '/checkpoints/DNF'
model_dir = os.path.join(save_dir, 'DNF_model')
log_dir = os.path.join(save_dir, 'DNF_loss')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def preprocess_audio(audio_path, target_length=16384, sr=8000):
    audio, _ = librosa.load(audio_path, sr=sr)
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    return audio

def load_data(directory, output_dir):
    filenames = {f: os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')}
    audio_data = []
    for filename, path in filenames.items():
        audio = preprocess_audio(path)
        audio_data.append(audio)
    audio_data = np.array(audio_data)
    np.save(os.path.join(output_dir, 'audio_data.npy'), audio_data)
    return audio_data

def global_median_pooling1d(inputs):

    return tfp.stats.percentile(inputs, 50.0, axis=1, keepdims=True)

class ChannelAttention1D(tf.keras.layers.Layer):
    def __init__(self, input_channels, reduction_ratio=4):
        super(ChannelAttention1D, self).__init__()
        self.input_channels = input_channels
        reduced_channels = max(1, input_channels // reduction_ratio)
        self.fc1 = tf.keras.layers.Conv1D(filters=reduced_channels, kernel_size=1)
        self.fc2 = tf.keras.layers.Conv1D(filters=input_channels, kernel_size=1)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=1, keepdims=True)
        median_pool = tfp.stats.percentile(inputs, 50.0, axis=1, keepdims=True, interpolation='midpoint')

        avg_out = tf.nn.relu(self.fc1(avg_pool))
        avg_out = tf.nn.sigmoid(self.fc2(avg_out))

        max_out = tf.nn.relu(self.fc1(max_pool))
        max_out = tf.nn.sigmoid(self.fc2(max_out))

        median_out = tf.nn.relu(self.fc1(median_pool))
        median_out = tf.nn.sigmoid(self.fc2(median_out))

        return avg_out + max_out + median_out

class SpatialAttention1D(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(SpatialAttention1D, self).__init__()
        self.channels = channels

        self.conv_layers = [
            tf.keras.layers.Conv1D(filters=channels, kernel_size=size, padding='same', groups=channels, activation='sigmoid')
            for size in [3, 5, 7, 9, 11]  
        ]
        self.combine_conv = tf.keras.layers.Conv1D(filters=channels, kernel_size=1, activation='sigmoid')

    def call(self, inputs):

        outputs = [conv(inputs) for conv in self.conv_layers]
        summed_output = tf.add_n(outputs)  
        return self.combine_conv(summed_output)

class Audio_DNF(tf.keras.Model): 
    def __init__(self, channels):
        super(Audio_DNF, self).__init__()
        self.channel_attention = ChannelAttention1D(input_channels=channels)
        self.spatial_attention = SpatialAttention1D(channels=channels)

    def call(self, inputs):
        channel_out = self.channel_attention(inputs) * inputs
        spatial_out = self.spatial_attention(channel_out) * inputs 
        return spatial_out

def train_DNF(model, processed_output, epochs):
    #All the code will be released after acceptance.

def train_combined():
    #All the code will be released after acceptance.

if __name__ == '__main__':
    train_combined()