import numpy as np
import os
import librosa
import tensorflow as tf
from scipy.io.wavfile import write
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial audio samples.")
    parser.add_argument("--frame_length", type=int, default=128, help="Frame length for processing.")
    parser.add_argument("--stride_length", type=int, default=128, help="Stride length for processing.")
    parser.add_argument("--model_dir", type=str, default="/audioattack/model", help="Directory containing model files.")
    parser.add_argument("--epsilon", type=float, default=0.002, help="Perturbation magnitude for FGSM.")
    parser.add_argument("--alpha", type=float, default=0.001, help="Step size for PGD.")
    parser.add_argument("--num_iter", type=int, default=40, help="Number of iterations for PGD.")
    return parser.parse_args()

args = parse_args()

sem_enc = tf.keras.models.load_model(os.path.join(args.model_dir, 'sem_enc.h5'))


def pad_or_trim(wav, length=16384):
    if len(wav) > length:
        return wav[:length]
    elif len(wav) < length:
        return np.pad(wav, (0, length - len(wav)), mode='constant')
    return wav

def fgsm_attack(model, input, epsilon):
    input_expanded = tf.expand_dims(input, axis=0)
    with tf.GradientTape() as tape:
        tape.watch(input_expanded)
        prediction = model(input_expanded)[0]
        prediction_flattened = tf.reduce_mean(prediction, axis=[1, 2, 3])
        loss = tf.keras.losses.mean_squared_error(input_expanded, prediction_flattened)
    gradient = tape.gradient(loss, input_expanded)
    adversarial_example = input_expanded + epsilon * tf.sign(gradient)
    adversarial_example = tf.squeeze(adversarial_example, axis=0)
    return adversarial_example

def process_directory(input_dir, output_dir, sample_rate=8000):
    #All the code will be released after acceptance.


input_directory = '/data'
output_directory = '/audioattack/outputaudio'

process_directory(input_directory, output_directory)
