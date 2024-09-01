import numpy as np
import os
import librosa
from scipy.io.wavfile import write
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial audio samples.")
    parser.add_argument("--sample_rate", type=int, default=8000, help="Sample rate for audio processing.")
    parser.add_argument("--max_length", type=int, default=16384, help="Maximum length for audio samples.")
    parser.add_argument("--mask_percentage", type=float, default=0.10, help="Percentage of the audio to mask.")
    return parser.parse_args()

args = parse_args()

def pad_or_trim(wav, length=16384):

    if len(wav) > length:
        return wav[:length]
    elif len(wav) < length:
        return np.pad(wav, (0, length - len(wav)), mode='constant')
    return wav

def random_mask(audio, mask_percentage=0.10):
    total_samples = len(audio)
    num_masked_samples = int(total_samples * mask_percentage)
    mask_indices = np.random.choice(total_samples, num_masked_samples, replace=False)
    audio[mask_indices] = 0
    return audio

def process_directory(input_dir, output_dir, sample_rate=8000, max_length=16384, mask_percentage=0.10):
    #All the code will be released after acceptance.

if __name__ == '__main__':
    input_directory = '/semantic/data/nowdata/clean_trainset_28spk_wav_8k'
    output_directory = '/audioattack/outputaudio/output_attack_mask_train'

    process_directory(input_directory, output_directory, args.sample_rate, args.max_length, args.mask_percentage)
