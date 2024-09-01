import tensorflow as tf
import numpy as np
import librosa
import os
import scipy.io.wavfile as wav

def load_generator(model_path):
    return tf.keras.models.load_model(model_path)

def load_and_process_audio(directory, sample_rate=8000, max_length=16384):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
    audios = []
    filenames = []
    for file in files:
        audio, sr = librosa.load(file, sr=sample_rate)
        audio = pad_or_trim(audio, max_length)
        audios.append(audio)
        filenames.append(os.path.basename(file)) 
    return audios, filenames

def pad_or_trim(audio, length=16384):
    if len(audio) > length:
        return audio[:length]
    elif len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), mode='constant')
    return audio

def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_audio(path, audio, sample_rate):
    audio_int16 = np.int16(audio * 32767)
    wav.write(path, sample_rate, audio_int16)

def generate_and_save_adversarial_samples(generator, original_audios, filenames, output_directory):
    #All the code will be released after acceptance.


if __name__ == '__main__':
    generator_path = '/AdvGAN/outdata/model/generator_epoch.h5'
    audio_directory = '/data'
    output_directory = '/AdvGAN/outputaudio'

    gen_model = load_generator(generator_path)
    original_audios, filenames = load_and_process_audio(audio_directory)
    generate_and_save_adversarial_samples(gen_model, original_audios, filenames, output_directory)