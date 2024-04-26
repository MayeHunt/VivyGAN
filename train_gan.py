import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input, Flatten, Conv2D, Activation, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import keras
from keras.utils import img_to_array, load_img
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from PIL import Image



def preprocess(file_path):
    img = load_img(file_path, color_mode='grayscale', target_size=(72, 384))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array


def create_dataset(csv_path, set_type, batch_size):
    df = pd.read_csv(csv_path)
    set_df = df[df['set'] == set_type]
    file_paths = set_df['file_path'].tolist()
    file_paths = [os.path.join(os.getcwd(), fp.replace('\\', '/')) for fp in file_paths]

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(lambda x: tf.numpy_function(preprocess, [x], tf.float32), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(lambda x: tf.reshape(x, (72, 384, 1)))

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_dataset_size(csv_path, set_type):
    df = pd.read_csv(csv_path)
    set_df = df[df['set'] == set_type]
    return len(set_df)


def generate_noise(batch_size, latent_dim):
    return np.random.normal(0, 1, (batch_size, latent_dim))


def build_generator(latent_dim=100):
    model = Sequential([
        Input(shape=(latent_dim,)),
        Dense(9 * 48 * 512, kernel_initializer='he_normal'),
        LeakyReLU(negative_slope=0.2),
        Reshape((9, 48, 512)),
        Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal'),
        LeakyReLU(negative_slope=0.2),
        Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal'),
        LeakyReLU(negative_slope=0.2),
        Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal'),
        LeakyReLU(negative_slope=0.2),
        Conv2DTranspose(1, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal'),
        Activation('sigmoid'),
    ])
    return model


def build_discriminator(seq_length=384):
    model = Sequential([
        Input(shape=(seq_length, 72, 1)),
        Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal'),
        LeakyReLU(negative_slope=0.2),
        Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal'),
        LeakyReLU(negative_slope=0.2),
        Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal'),
        LeakyReLU(negative_slope=0.2),
        Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal'),
        LeakyReLU(negative_slope=0.2),
        Flatten(),
        Dense(1, kernel_initializer='he_normal')
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


def generator_loss(fake_output, generated_images, density_weight=0.02, density_threshold=7, silence_weight=0.1, fragmentation_weight=0.001):
    adversarial_loss = -tf.reduce_mean(fake_output)
    density_penalty = calculate_density_penalty(generated_images, density_threshold)
    silence_penalty = calculate_silence_penalty(generated_images)
    fragmentation_penalty = calculate_fragmentation_penalty(generated_images)

    combined_loss = adversarial_loss + (density_weight * density_penalty) + (silence_penalty * silence_weight) + (fragmentation_penalty * fragmentation_weight)
    return combined_loss, (density_penalty*density_weight), (silence_penalty*silence_weight), (fragmentation_penalty*fragmentation_weight)


def gradient_penalty(real_images, fake_images, discriminator, batch_size):
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


def calculate_density_penalty(generated_images, threshold):
    active_notes = tf.reduce_sum(generated_images, axis=1)
    excessive_density = tf.maximum(active_notes - threshold, 0)
    density_penalty = tf.reduce_sum(excessive_density)
    return density_penalty


def calculate_silence_penalty(generated_images):
    silent_timesteps = tf.reduce_sum(tf.cast(tf.reduce_sum(generated_images, axis=1) == 0, tf.float32))
    return silent_timesteps


def calculate_fragmentation_penalty(generated_images, note_duration_threshold=12):
    note_changes = generated_images[:, :, 1:, :] - generated_images[:, :, :-1, :]
    note_endings = tf.cast(tf.equal(note_changes, -1), tf.float32)
    cumulative_active_notes = tf.cumsum(generated_images, axis=2)
    reset_at_inactive = cumulative_active_notes * generated_images
    shifted_active_notes = tf.pad(
        reset_at_inactive[:, :, :-1, :],
        paddings=((0, 0), (0, 0), (1, 0), (0, 0)),
        mode="CONSTANT",
        constant_values=0
    )
    distance_to_threshold = note_duration_threshold - shifted_active_notes[:, :, 1:]
    smooth_penalty = tf.nn.sigmoid(distance_to_threshold) * note_endings
    smooth_penalty = tf.maximum(smooth_penalty, 0)
    scaled_penalty = smooth_penalty * 10
    penalty = tf.reduce_sum(scaled_penalty)
    return penalty


@tf.function
def train_discriminator_step(images, generator, discriminator, batch_size, latent_dim, gp_weight=20.0):
    noise = generate_noise(batch_size, latent_dim)

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=False)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gp = gradient_penalty(images, generated_images, discriminator, batch_size)
        disc_loss = discriminator_loss(real_output, fake_output) + (gp * gp_weight)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return disc_loss, tf.reduce_mean(real_output), gp


@tf.function
def train_generator_step(generator, discriminator, batch_size, latent_dim):
    noise = generate_noise(batch_size, latent_dim)

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=False)
        gen_loss, density_penalty, silence_penalty, fragmentation_penalty = generator_loss(fake_output, generated_images)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gen_loss, density_penalty, silence_penalty, fragmentation_penalty


def train_gan(generator, discriminator, train_dataset, epochs, batch_size, latent_dim, steps_per_epoch, save_interval=10, critic_updates=3):
    os.makedirs('generated', exist_ok=True)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        gen_loss_list, disc_loss_list = [], []
        batch_count = 1
        discriminator_update_counter = 0

        for image_batch in train_dataset:

            if discriminator_update_counter < critic_updates:
                # Discriminator update phase
                discriminator.trainable = True
                disc_loss, real_images_loss, gp = train_discriminator_step(image_batch, generator, discriminator, batch_size, latent_dim)
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_count}/{steps_per_epoch}, Discriminator Loss: {disc_loss:.2f}, Real Loss: {real_images_loss:.2f}, Gradient Penalty: {gp:.2f}")
                disc_loss_list.append(disc_loss)
                discriminator_update_counter += 1
            else:
                # Generator update phase
                gen_loss, dp, sp, fp = train_generator_step(generator, discriminator, batch_size, latent_dim)
                gen_loss_list.append(gen_loss)

                discriminator_update_counter = 0

                print(
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_count}/{steps_per_epoch}, "
                    f"Avg Discriminator Loss: {np.mean(disc_loss_list[-critic_updates:]):.2f}, Generator Loss: {gen_loss:.2f}, Density Penalty: {dp:.2f}, Silence Penalty: {sp:.2f}, Fragmentation Penalty: {fp:.2f}")

            # TensorBoard logging
            with train_summary_writer.as_default():
                if 'gen_loss' in locals():
                    tf.summary.scalar('Generator Loss', gen_loss, step=epoch * steps_per_epoch + batch_count)
                if 'disc_loss' in locals():
                    tf.summary.scalar('Discriminator Loss', disc_loss, step=epoch * steps_per_epoch + batch_count)
                if 'real_output_mean' in locals():
                    tf.summary.scalar('Real Output Mean', real_images_loss, step=epoch * steps_per_epoch + batch_count)
                if 'gp' in locals():
                    tf.summary.scalar('Gradient Penalty', gp, step=epoch * steps_per_epoch + batch_count)
                if 'dp' in locals():
                    tf.summary.scalar('Density Penalty', dp, step=epoch * steps_per_epoch + batch_count)
                if 'sp' in locals():
                    tf.summary.scalar('Silence Penalty', sp, step=epoch * steps_per_epoch + batch_count)
                if 'fp' in locals():
                    tf.summary.scalar('Fragmentation Penalty', fp, step=epoch * steps_per_epoch + batch_count)

            batch_count += 1


            if batch_count >= steps_per_epoch:
                break

        if epoch == 0:
            save_imgs(generator, epoch, latent_dim)
            for real_images in train_dataset.take(1):
                real_images_np = real_images.numpy()
                plot_real_vs_generated(generator, real_images_np, epoch, latent_dim=latent_dim, examples=5)
                break

        if (epoch + 1) % save_interval == 0:
            manager.save()
            save_imgs(generator, epoch, latent_dim)
            for real_images in train_dataset.take(1):
                real_images_np = real_images.numpy()
                plot_real_vs_generated(generator, real_images_np, epoch, latent_dim=latent_dim, examples=5)
                break

        total_epoch_time = (time.perf_counter() - epoch_start)
        print_epoch_summary(epoch, epochs, gen_loss_list, disc_loss_list, total_epoch_time)

def print_epoch_summary(epoch, epochs, gen_loss_list, disc_loss_list, total_epoch_time):
    print(f"--- Epoch {epoch+1}/{epochs} Summary ---")
    print(f"Avg Generator Loss: {np.mean(gen_loss_list):.2f}, Avg Discriminator Loss: {np.mean(disc_loss_list):.2f}")
    print(f"Total Epoch Time: {total_epoch_time:.2f} seconds")


def save_imgs(generator, epoch, latent_dim=10, examples=1):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)

    for i, img in enumerate(generated_images):
        img = np.squeeze(img)

        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img.astype(np.uint8)

        img = Image.fromarray(img, 'L')
        img.save(f'generated/epoch_{epoch}_example_{i}.png')

    print(f"Generated images for epoch {epoch} saved.")


def plot_real_vs_generated(generator, real_data, epoch, latent_dim=100, examples=10):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2

    real_data_subset = (real_data[:examples] + 1) / 2

    fig, axs = plt.subplots(examples, 2, figsize=(12, 4))

    for i in range(examples):
        axs[i, 0].imshow(real_data_subset[i].squeeze(), cmap='gray', aspect='equal', interpolation='none')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(generated_images[i].squeeze(), cmap='gray', aspect='equal', interpolation='none')
        axs[i, 1].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f'generated/real_vs_generated_epoch_{epoch}.png')
    plt.close(fig)
    print(f"Real vs Generated images saved to 'generated/real_vs_generated_epoch_{epoch}.png'")


def model_checkpoints(checkpoint_dir, generator, discriminator):
    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored model weights from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")


if __name__ == "__main__":
    # Params
    batch_size = 128
    save_interval = 10
    latent_dim = 100
    epochs = 600
    csv_path = 'dataset_split.csv'

    # Dataset
    train_dataset = create_dataset(csv_path, 'train', batch_size)
    validation_dataset = create_dataset(csv_path, 'validation', batch_size)
    test_dataset = create_dataset(csv_path, 'test', batch_size)
    # Dataset size metrics
    train_size = get_dataset_size(csv_path, 'train')
    validation_size = get_dataset_size(csv_path, 'validation')
    test_size = get_dataset_size(csv_path, 'test')
    # Used for limiting loops
    steps_per_epoch = train_size // batch_size
    validation_steps = validation_size // batch_size

    # Building Models
    generator = build_generator()
    discriminator = build_discriminator()
    # Optimisers
    generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999)
    discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999)
    # Compiled GAN
    gan = build_gan(generator, discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

    # TF Checkpoints
    checkpoint_dir = './training_checkpoints_cnn'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored model weights from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    # TensorBoard
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    train_log_dir = './logs/gradient_tape/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Save model as .keras
    generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
    gan.save('model/model_save.keras')
    generator.save('model/generator_save.keras')
    # Load model
    # model = keras.models.load_model('model/model_save.keras')


    train_gan(generator, discriminator, train_dataset, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim,
              steps_per_epoch=train_size // batch_size)
