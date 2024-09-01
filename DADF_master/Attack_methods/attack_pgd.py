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
    parser.add_argument("--epsilon", type=float, default=0.002, help="Perturbation magnitude for PGSM.")
    parser.add_argument("--alpha", type=float, default=0.001, help="Step size for PGD.")
    parser.add_argument("--num_iter", type=int, default=40, help="Number of iterations for PGD.")
    return parser.parse_args()

args = parse_args()


sem_enc = tf.keras.models.load_model(os.path.join(args.model_dir, 'sem_enc.h5'))

def pgd_attack(model, input, epsilon, alpha, num_iter):
    input_expanded = tf.expand_dims(input, axis=0)
    adversarial_example = input_expanded
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_example)
            prediction = model(adversarial_example)

            if isinstance(prediction, list):
                prediction = prediction[0]

            prediction_reduced = tf.reduce_mean(prediction, axis=[1, 2, 3], keepdims=True)
            prediction_resized = tf.repeat(prediction_reduced, repeats=input_expanded.shape[1], axis=1)
            prediction_flat = tf.reshape(prediction_resized, [1, -1])
            input_flat = tf.reshape(input_expanded, [1, -1])

            loss = tf.keras.losses.mean_squared_error(input_flat, prediction_flat)

        gradient = tape.gradient(loss, adversarial_example)
        adversarial_example += alpha * tf.sign(gradient)
        perturbation = tf.clip_by_value(adversarial_example - input_expanded, -epsilon, epsilon)
        adversarial_example = input_expanded + perturbation
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
