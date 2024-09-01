import tensorflow as tf

def discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(16384, 1)),
        tf.keras.layers.Conv1D(16, 25, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1D(32, 25, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1D(64, 25, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    return model