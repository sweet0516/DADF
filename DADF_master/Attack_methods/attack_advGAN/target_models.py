import tensorflow as tf
import numpy as np
import os
import librosa


class Target:
    def __init__(self, model_path=None, speaker_id_map=None):
        self.model_path = model_path
        self.speaker_id_map = speaker_id_map
        if self.model_path and os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.model = self.build_model()

    def load_audio_data(self, input_dir, sample_rate=8000, max_length=16384):
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.wav')]
        data = []
        labels = []

        for file_path in files:
            audio, _ = librosa.load(file_path, sr=sample_rate)
            if len(audio) > max_length:
                audio = audio[:max_length]
            elif len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            data.append(audio)
            filename = os.path.basename(file_path)
            speaker_id = filename.split('_')[0]
            if speaker_id in self.speaker_id_map:
                label = self.speaker_id_map[speaker_id]
                labels.append(label)
            else:
                continue

        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return data, labels

    def build_model(self):
        if self.speaker_id_map is None:
            raise ValueError("speaker_id_map must be defined to build the model")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(16384, 1)),
            tf.keras.layers.Conv1D(32, 3, padding="same", activation='relu'),
            tf.keras.layers.Conv1D(32, 3, padding="same", activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, padding="same", activation='relu'),
            tf.keras.layers.Conv1D(64, 3, padding="same", activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(len(self.speaker_id_map), activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, Y, X_test, Y_test):
        X = np.expand_dims(X, -1)
        X_test = np.expand_dims(X_test, -1)
        self.model.fit(X, Y, epochs=50, batch_size=16, validation_data=(X_test, Y_test))
        # Save the trained model
        model_save_path = '/AdvGAN/model'
        self.model.save(os.path.join(model_save_path, 'trained_model.h5'))

if __name__ == '__main__':
    speaker_id_map_train = {'p226': 0, 'p227': 1, 'p228': 2, 'p230': 3, 'p231': 4, 'p233': 5, 'p236': 6, 'p239': 7, 'p243': 8, 'p244': 9, 'p250': 10, 'p254': 11, 'p256': 12, 'p258': 13, 'p259': 14, 'p267': 15, 'p268': 16, 'p269': 17, 'p270': 18, 'p273': 19, 'p274': 20, 'p276': 21, 'p277': 22, 'p278': 23, 'p279': 24, 'p282': 25, 'p286': 26, 'p287': 27}
    speaker_id_map_test = {'p232': 0, 'p257': 1}

    target_model_train = Target(speaker_id_map=speaker_id_map_train)
    target_model_test = Target(speaker_id_map=speaker_id_map_test)

    training_data_directory = '/data/trainset'
    testing_data_directory = '/data/testset'

    X_train, y_train = target_model_train.load_audio_data(training_data_directory)
    X_test, y_test = target_model_test.load_audio_data(testing_data_directory)

    target_model_train.train(X_train, y_train, X_test, y_test)