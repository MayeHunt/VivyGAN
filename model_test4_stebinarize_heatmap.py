import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape, TimeDistributed, LeakyReLU, BatchNormalization, Input, Conv1D, Conv1DTranspose, Flatten, Conv2D, Activation, Layer, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import pypianoroll
from pypianoroll import *

def preprocess(file_path):
    def load_npz(file_path):
        data = np.load(file_path)
        piano_roll = data['piano_roll']
        return np.array(piano_roll, dtype=np.float32)

    piano_roll = tf.numpy_function(load_npz, [file_path], tf.float32)
    piano_roll.set_shape((96, 72))
    return piano_roll


def create_dataset(csv_path, set_type, batch_size):
    df = pd.read_csv(csv_path)
    set_df = df[df['set'] == set_type]
    file_paths = set_df['file_path'].tolist()
    file_paths = [os.path.join(os.getcwd(), fp.replace('\\', '/')) for fp in file_paths]

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(lambda x: preprocess(x), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class STEBinarize(Layer):
    def __init__(self, initial_temperature=5.0, **kwargs):
        super(STEBinarize, self).__init__(**kwargs)
        self.initial_temperature = initial_temperature
        self.temperature = tf.Variable(initial_temperature, trainable=False, dtype=tf.float32)

    def call(self, x):
        return tf.math.sigmoid(self.temperature * x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(STEBinarize, self).get_config()
        config.update({
            'initial_temperature': self.initial_temperature
        })
        return config


def get_dataset_size(csv_path, set_type):
    df = pd.read_csv(csv_path)
    set_df = df[df['set'] == set_type]
    return len(set_df)


def generate_noise(batch_size, latent_dim):
    return np.random.normal(0, 1, (batch_size, latent_dim))


def build_generator(latent_dim=100):
    model = Sequential([
        Input(shape=(latent_dim,)),
        Dense(96 * 72),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Reshape((24, 18, 16)),
        Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(1, kernel_size=(3, 3), strides=(2, 2), padding='same'),  # Adjust for final layer
        Activation('tanh'),
        STEBinarize(),
    ])
    return model


def build_discriminator(seq_length=96):
    model = Sequential([
        Input(shape=(seq_length, 72, 1)),
        Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images, generator, discriminator, gan, batch_size, latent_dim):
    noise = generate_noise(batch_size, latent_dim)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
        real_accuracy = tf.reduce_mean(tf.cast(tf.math.greater_equal(real_output, 0.5), tf.float32))
        fake_accuracy = tf.reduce_mean(tf.cast(tf.math.less(fake_output, 0.5), tf.float32))

    discriminator.trainable = True
    for layer in discriminator.layers:
        assert layer.trainable, f"Layer {layer.name} is not trainable but should be."

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    if not gradients_of_discriminator or any(g is None for g in gradients_of_discriminator):
        print("Discriminator gradients are empty or contain None values.")
    else:
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, real_accuracy, fake_accuracy


def train_gan(generator, discriminator, gan, train_dataset, epochs, batch_size, latent_dim, steps_per_epoch,
              save_interval=10, initial_temperature=5.0, final_temperature=0.1):
    os.makedirs('generated', exist_ok=True)
    
    ste_bin_layer = [layer for layer in generator.layers if isinstance(layer, STEBinarize)][0]
    temperature_decrement = (initial_temperature - final_temperature) / epochs

    for epoch in range(epochs):
        epoch_start = time.time()  
        # train_dataset = train_dataset.repeat()
        gen_loss_list = []
        disc_loss_list = []
        real_acc_list = []
        fake_acc_list = []
        batch_count = 0
        
        new_temperature = max(ste_bin_layer.temperature - temperature_decrement, final_temperature)
        tf.keras.backend.set_value(ste_bin_layer.temperature, new_temperature)

        for image_batch in train_dataset:
            start_time = time.time()

            # image_batch = tf.transpose(image_batch, perm=[0, 2, 1])
            gen_loss, disc_loss, real_accuracy, fake_accuracy = train_step(image_batch, generator, discriminator, gan, batch_size, latent_dim)
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)
            real_acc_list.append(real_accuracy)
            fake_acc_list.append(fake_accuracy)

            end_time = time.time()
            batch_time = (end_time - start_time) * 1000

            batch_count += 1
            print(
                f'Epoch {epoch + 1}/{epochs}, Batch {batch_count}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}, Real Acc: {real_accuracy:.2f}, Fake Acc: {fake_accuracy:.2f}, Time: {batch_time:.2f} ms')

            if batch_count >= steps_per_epoch:
                break

        epoch_end = time.time()
        total_epoch_time = epoch_end - epoch_start

        gen_loss_avg = np.mean(gen_loss_list)
        disc_loss_avg = np.mean(disc_loss_list)
        real_acc_avg = np.mean(real_acc_list)
        fake_acc_avg = np.mean(fake_acc_list)

        print(
            f'--- Epoch Summary: {epoch + 1}/{epochs}, Avg Gen Loss: {gen_loss_avg:.4f}, Avg Disc Loss: {disc_loss_avg:.4f}, Avg Real Acc: {real_acc_avg:.2f}, Avg Fake Acc: {fake_acc_avg:.2f}, Total Epoch Time: {total_epoch_time:.2f} seconds ---')

        if (epoch + 1) == 1:
            save_imgs(generator, epoch, latent_dim=100)

        if (epoch + 1) % save_interval == 0:
            manager.save()
            print(f'Saving model and generated images at epoch {epoch + 1}')
            save_imgs(generator, epoch, latent_dim=100)


def save_imgs(generator, epoch, latent_dim=10, examples=10, figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    # print(generated_images.shape)
    
    generated_images = (generated_images + 1) / 2

    generated_images = generated_images.reshape(examples, 96, 72)

    plt.figure(figsize=figsize)
    plt.imshow(generated_images[0].T, interpolation='nearest', cmap='gray_r')
    plt.axis('off')
    plt.savefig(f'generated/epoch_{epoch}.png')
    plt.close()
    print(f"Generated images saved to 'generated/epoch_{epoch}.png'")


batch_size = 32
save_interval = 10
csv_path = 'sampled_dataset_split.csv'
train_dataset = create_dataset(csv_path, 'train', batch_size)
validation_dataset = create_dataset(csv_path, 'validation', batch_size)
test_dataset = create_dataset(csv_path, 'test', batch_size)

train_size = get_dataset_size(csv_path, 'train')
validation_size = get_dataset_size(csv_path, 'validation')
test_size = get_dataset_size(csv_path, 'test')

steps_per_epoch = train_size // batch_size
validation_steps = validation_size // batch_size

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-5)

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

checkpoint_dir = './training_checkpoints_cnn5'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)

manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f"Restored model weights from {manager.latest_checkpoint}")
else:
    print("Initializing from scratch.")

train_gan(generator, discriminator, gan, train_dataset, epochs=100, batch_size=batch_size, latent_dim=100,
          steps_per_epoch=train_size // batch_size)
