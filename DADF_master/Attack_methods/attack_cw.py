import numpy as np
import os
import librosa
import tensorflow as tf
from scipy.io.wavfile import write
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial audio samples.")
    parser.add_argument("--model_dir", type=str, default="/audioattack/model", help="Directory containing model files.")
    parser.add_argument("--conf", type=float, default=0, help="Confidence of the attack.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of iterations for the optimization.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--target_class", type=int, default=None, help="Target class for misclassification (if None, untargeted).")
    return parser.parse_args()

args = parse_args()

sem_enc = tf.keras.models.load_model(os.path.join(args.model_dir, 'sem_enc.h5'))

def cw_attack(model, input_tensor, target_class, conf, learning_rate, max_iterations):
    input_expanded = tf.expand_dims(input_tensor, axis=0)
    output = model(input_expanded)

    if isinstance(output, list):
        output = output[0]

    delta = tf.Variable(tf.zeros_like(input_tensor), trainable=True)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    for i in range(max_iterations):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            new_input = input_expanded + delta
            output = model(new_input)
            if isinstance(output, list):
                output = output[0]

            if target_class is not None:
                target_label = tf.one_hot(target_class, depth=output.shape[-1])
                target_output = tf.reduce_max(output * target_label, axis=1)
                loss = tf.maximum(0.0, target_output - tf.reduce_max(output, axis=1) + conf)
            else:
                current_max_output = tf.reduce_max(output, axis=1)
                next_max_output = tf.sort(output, axis=1, direction='DESCENDING')[:,1]
                loss = tf.maximum(0.0, next_max_output - current_max_output + conf)

            l2dist = tf.reduce_sum(tf.square(delta))

        gradient = tape.gradient([loss, l2dist], delta)
        optimizer.apply_gradients([(gradient, delta)])

        if tf.reduce_all(loss <= 0):
            break

    adversarial_example = tf.squeeze(input_expanded + delta, axis=0)
    return adversarial_example

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
