import tensorflow as tf

def generator():

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(16384, 1)), 
        tf.keras.layers.Conv1D(16, 25, strides=1, padding='same', activation='relu'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1D(32, 25, strides=2, padding='same', activation='relu'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1D(64, 25, strides=2, padding='same', activation='relu'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1DTranspose(32, 25, strides=2, padding='same', activation='relu'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1DTranspose(16, 25, strides=2, padding='same', activation='relu'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1D(1, 25, strides=1, padding='same', activation='tanh')
    ])

    return model