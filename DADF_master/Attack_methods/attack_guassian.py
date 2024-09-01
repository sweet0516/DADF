import numpy as np
import os
import librosa
from scipy.io.wavfile import write

def load_audio(file_path, sample_rate=8000, max_length=16384):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    if len(audio) > max_length:
        audio = audio[:max_length]
    elif len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
    return audio

def add_gaussian_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return np.clip(augmented_audio, -1., 1.)

def process_directory(input_dir, output_dir, sample_rate=8000):
    #All the code will be released after acceptance.
    
if __name__ == '__main__':
    input_directory = '/data'
    output_directory = '/audioattack/outputaudio'
    process_directory(input_directory, output_directory)
