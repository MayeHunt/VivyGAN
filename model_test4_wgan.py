import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Flatten, Conv2D, Activation, Layer, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time


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


def get_dataset_size(csv_path, set_type):
    df = pd.read_csv(csv_path)
    set_df = df[df['set'] == set_type]
    return len(set_df)


def plot_real_vs_generated(generator, real_data, epoch, latent_dim=100, examples=10, figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2

    real_data_subset = real_data[:examples]

    fig, axs = plt.subplots(2, examples, figsize=figsize)

    for i in range(examples):
        axs[0, i].imshow(real_data_subset[i].squeeze(), cmap='gray_r', aspect='auto')
        axs[0, i].axis('off')
        axs[1, i].imshow(generated_images[i].squeeze(), cmap='gray_r', aspect='auto')
        axs[1, i].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f'generated/real_vs_generated_epoch_{epoch}.png')
    plt.close()
    print(f"Real vs Generated images saved to 'generated/real_vs_generated_epoch_{epoch}.png'")


def generate_noise(batch_size, latent_dim):
    return np.random.normal(0, 1, (batch_size, latent_dim))


def build_generator(latent_dim=100):
    model = Sequential([
        Input(shape=(latent_dim,)),
        Dense(24 * 18 * 256),
        LeakyReLU(negative_slope=0.2),
        # BatchNormalization(momentum=0.8),
        Reshape((24, 18, 256)),
        Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        # BatchNormalization(momentum=0.8),
        Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        # BatchNormalization(momentum=0.8),
        Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        Activation('linear'),
    ])
    return model


def build_discriminator(seq_length=96, num_kernels=50, kernel_dim=5):
    model = Sequential([
        Input(shape=(seq_length, 72, 1)),
        Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        Flatten(),
        # MinibatchDiscrimination(num_kernels=num_kernels, kernel_dim=kernel_dim),
        Dense(1)
    ])
    return model


def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def gradient_penalty(real_images, fake_images, discriminator, batch_size):
    # Ensure the channel dimension is present in real_images
    real_images = tf.expand_dims(real_images, axis=-1) if real_images.shape[-1] != 1 else real_images

    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.) ** 2)
    return gp


@tf.function
def train_discriminator_step(images, generator, discriminator, batch_size, latent_dim, gp_weight=20.0):
    noise = generate_noise(batch_size, latent_dim)

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gp = gradient_penalty(images, generated_images, discriminator, batch_size)
        disc_loss = discriminator_loss(real_output, fake_output) + gp * gp_weight

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return disc_loss


@tf.function
def train_generator_step(generator, discriminator, batch_size, latent_dim):
    noise = generate_noise(batch_size, latent_dim)

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gen_loss


def train_gan(generator, discriminator, train_dataset, epochs, batch_size, latent_dim, steps_per_epoch, save_interval=10, critic_updates=5):
    os.makedirs('generated', exist_ok=True)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        gen_loss_list, disc_loss_list = [], []
        batch_count = 0
        update_counter = 0

        for image_batch in train_dataset:
            if update_counter % critic_updates == 0:
                for _ in range(critic_updates):
                    discriminator.trainable = True
                    disc_loss = train_discriminator_step(image_batch, generator, discriminator, batch_size, latent_dim)
                    disc_loss_list.append(disc_loss)

                gen_loss = train_generator_step(generator, discriminator, batch_size, latent_dim)
                gen_loss_list.append(gen_loss)
            else:
                gen_loss = train_generator_step(generator, discriminator, batch_size, latent_dim)
                gen_loss_list.append(gen_loss)

            batch_count += 1
            update_counter += 1

            if batch_count % 1 == 0 or batch_count == 1 or batch_count == steps_per_epoch:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}/{steps_per_epoch}, Generator Loss: {gen_loss:.4f}, Discriminator Loss: {np.mean(disc_loss_list[-critic_updates:]):.4f}")

            if batch_count >= steps_per_epoch:
                break

        total_epoch_time = (time.perf_counter() - epoch_start)

        if (epoch + 1) % save_interval == 0:
            manager.save()
            save_imgs(generator, epoch, latent_dim)
            for real_images in train_dataset.take(1):
                real_images_np = real_images.numpy()
                plot_real_vs_generated(generator, real_images_np, epoch, latent_dim=latent_dim, examples=10)
                break

        print_epoch_summary(epoch, epochs, gen_loss_list, disc_loss_list, total_epoch_time)

def print_epoch_summary(epoch, epochs, gen_loss_list, disc_loss_list, total_epoch_time):
    print(f"--- Epoch {epoch+1}/{epochs} Summary ---")
    print(f"Avg Generator Loss: {np.mean(gen_loss_list):.4f}, Avg Discriminator Loss: {np.mean(disc_loss_list):.4f}")
    print(f"Total Epoch Time: {total_epoch_time:.2f} seconds")


def save_imgs(generator, epoch, latent_dim=10, examples=10, figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    # print(generated_images.shape)

    generated_images = (generated_images + 1) / 2

    generated_images = generated_images.reshape(examples, 96, 72)

    plt.figure(figsize=figsize)
    plt.imshow(generated_images[0].T, interpolation='none', cmap='gray_r')
    plt.axis('off')
    plt.savefig(f'generated/epoch_{epoch}.png')
    plt.close()
    print(f"Generated images saved to 'generated/epoch_{epoch}.png'")


batch_size = 128
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

generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999)


gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

# TF Checkpoints
checkpoint_dir = './training_checkpoints_cnn'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f"Restored model weights from {manager.latest_checkpoint}")
else:
    print("Initializing from scratch.")

train_gan(generator, discriminator, train_dataset, epochs=200, batch_size=batch_size, latent_dim=100,
          steps_per_epoch=train_size // batch_size)
