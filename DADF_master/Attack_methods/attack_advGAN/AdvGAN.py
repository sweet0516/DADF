import tensorflow as tf
import numpy as np
import librosa
import os
import json
import matplotlib.pyplot as plt
from generator import generator
from discriminator import discriminator
from target_models import Target


def load_audio(directory, sample_rate=8000, max_length=16384):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
    audios = []
    for file in files:
        audio, sr = librosa.load(file, sr=sample_rate)
        if len(audio) > max_length:
            audio = audio[:max_length]
        elif len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
        audios.append(audio)
    audios = np.array(audios)
    return audios

def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_losses(g_losses, d_losses, epoch, path):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title(f"Losses at Epoch {epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{path}/loss_epoch_{epoch}.png")
    plt.close()

def AdvGAN_audio(directory, model_path, epochs=50, batch_size=128, save_interval=10):
    audio_samples = load_audio(directory)
    audio_samples = np.expand_dims(audio_samples, -1)

    target = Target(model_path=model_path)

    dataset = tf.data.Dataset.from_tensor_slices(audio_samples).batch(batch_size)

    gen_model = generator()
    disc_model = discriminator()

    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        for batch_x in dataset:
            z = np.random.normal(0, 1, size=batch_x.shape)
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                perturbed = gen_model(z, training=True)
                x_perturbed = perturbed + batch_x

                real_output = disc_model(batch_x, training=True)
                fake_output = disc_model(x_perturbed, training=True)

                d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output)))
                d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output)))
                d_loss = d_loss_real + d_loss_fake

                g_loss = -tf.reduce_mean(fake_output)

                gradients_of_generator = g_tape.gradient(g_loss, gen_model.trainable_variables)
                gradients_of_discriminator = d_tape.gradient(d_loss, disc_model.trainable_variables)

                g_optimizer.apply_gradients(zip(gradients_of_generator, gen_model.trainable_variables))
                d_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_model.trainable_variables))

            g_losses.append(g_loss.numpy())
            d_losses.append(d_loss.numpy())

        print(f"Epoch {epoch+1}, Gen Loss: {g_loss.numpy()}, Disc Loss: {d_loss.numpy()}")

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            loss_dir = '/AdvGAN/outdata/loss'
            make_sure_path_exists(loss_dir)
            plot_losses(g_losses, d_losses, epoch + 1, loss_dir)

            model_dir = '/AdvGAN/outdata/model'
            samples_dir = '/AdvGAN/outdata/samples'
            make_sure_path_exists(model_dir)
            make_sure_path_exists(samples_dir)
            gen_model.save(f'{model_dir}/generator_epoch_{epoch+1}.h5')
            disc_model.save(f'{model_dir}/discriminator_epoch_{epoch+1}.h5')
            np.save(f'{samples_dir}/generated_samples_epoch_{epoch+1}.npy', gen_model(z, training=False).numpy())

    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'learning_rate': 'default',
        'save_interval': save_interval
    }
    with open('training_config.json', 'w') as f:
        json.dump(config, f)

if __name__ == '__main__':
    audio_dir = '/data/trainset'
    model_path = '/model/trained_model.h5'
    AdvGAN_audio(audio_dir, model_path)