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
    parser.add_argument("--epsilon", type=float, default=0.002, help="Perturbation magnitude for BIM.")
    parser.add_argument("--alpha", type=float, default=0.001, help="Step size for each BIM iteration.")
    parser.add_argument("--num_iter", type=int, default=40, help="Number of iterations for BIM.")
    return parser.parse_args()

args = parse_args()

sem_enc = tf.keras.models.load_model(os.path.join(args.model_dir, 'sem_enc.h5'))

def bim_attack(model, input_tensor, epsilon, alpha, num_iter):
    input_expanded = tf.expand_dims(input_tensor, axis=0)
    adversarial_example = input_expanded
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_example)
            prediction = model(adversarial_example)[0]
            prediction_flattened = tf.reduce_mean(prediction, axis=[1, 2, 3])
            loss = tf.keras.losses.mean_squared_error(input_expanded, prediction_flattened)
        gradient = tape.gradient(loss, adversarial_example)
        adversarial_example = adversarial_example + alpha * tf.sign(gradient)
        adversarial_example = tf.clip_by_value(adversarial_example, input_expanded - epsilon, input_expanded + epsilon)
    return tf.squeeze(adversarial_example, axis=0)

def pad_or_trim(wav, length=16384):
    if len(wav) > length:
        return wav[:length]
    elif len(wav) < length:
        return np.pad(wav, (0, length - len(wav)), mode='constant')
    return wav

def process_directory(input_dir, output_dir, sample_rate=8000):
    #All the code will be released after acceptance.

input_directory = '/data'
output_directory = '/audioattack/outputaudio'

process_directory(input_directory, output_directory)
