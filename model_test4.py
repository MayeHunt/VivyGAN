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


def preprocess(file_path):
    def load_npz(file_path):
        data = np.load(file_path)
        piano_roll = data['piano_roll']
        return np.array(piano_roll, dtype=np.float32)

    piano_roll = tf.numpy_function(load_npz, [file_path], tf.float32)
    piano_roll.set_shape((96, 72))
    return piano_roll


class MinibatchDiscrimination(Layer):
    def __init__(self, num_kernels, kernel_dim, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim

    def build(self, input_shape):
        self.W = self.add_weight(name='weights',
                                 shape=(input_shape[-1], self.num_kernels * self.kernel_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, x):
        minibatch_features = tf.matmul(x, self.W)
        minibatch_features = tf.reshape(minibatch_features, (-1, self.num_kernels, self.kernel_dim))

        diffs = tf.expand_dims(minibatch_features, 3) - \
                tf.expand_dims(tf.transpose(minibatch_features, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), axis=2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), axis=2)

        return tf.concat([x, minibatch_features], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.num_kernels)


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
        Dense(24 * 18 * 512),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Reshape((24, 18, 512)),
        Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        Activation('sigmoid'),
    ])
    return model


def build_discriminator(seq_length=96, num_kernels=50, kernel_dim=5):
    input_layer = Input(shape=(seq_length, 72, 1))

    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x_flattened = Flatten()(x)

    x = MinibatchDiscrimination(num_kernels=num_kernels, kernel_dim=kernel_dim)(x_flattened)

    final_output = Dense(1, activation='sigmoid')(x)

    discriminator_model = tf.keras.Model(inputs=input_layer, outputs=final_output)
    discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    feature_extractor_model = tf.keras.Model(inputs=input_layer, outputs=discriminator_model.layers[2].output)

    return discriminator_model, feature_extractor_model


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
def train_step(images, generator, discriminator, feature_extractor, batch_size, latent_dim, update_discriminator, lambda_fm=0.1):
    noise = generate_noise(batch_size, latent_dim)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        if update_discriminator:
            real_output = discriminator(images, training=True)
            disc_loss = discriminator_loss(real_output, fake_output)

            real_features = feature_extractor(images, training=False)
            fake_features = feature_extractor(generated_images, training=False)
            feature_matching_loss = tf.reduce_mean(tf.abs(real_features - fake_features))

            real_accuracy = tf.reduce_mean(tf.cast(tf.math.greater_equal(real_output, 0.5), tf.float32))
            fake_accuracy = tf.reduce_mean(tf.cast(tf.math.less(fake_output, 0.5), tf.float32))
        else:
            disc_loss = 0
            feature_matching_loss = 0
            real_accuracy = 0
            fake_accuracy = 0
        gen_loss = generator_loss(fake_output) + lambda_fm * feature_matching_loss

    discriminator.trainable = True

    for layer in discriminator.layers:
        assert layer.trainable, f"Layer {layer.name} is not trainable but should be."

    if update_discriminator:
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        if not gradients_of_discriminator or any(g is None for g in gradients_of_discriminator):
            print("Discriminator gradients are empty or contain None values.")
        else:
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss, real_accuracy, fake_accuracy, feature_matching_loss


def train_gan(generator, discriminator, feature_extractor, train_dataset, epochs, batch_size, latent_dim, steps_per_epoch,
              save_interval=10):
    os.makedirs('generated', exist_ok=True)
    discriminator_update_counter = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        gen_loss_list, disc_loss_list, real_acc_list, fake_acc_list = [], [], [], []
        batch_count = 0

        for image_batch in train_dataset:
            start_time = time.perf_counter()
            update_discriminator = (discriminator_update_counter % 1 == 0)

            gen_loss, disc_loss, real_accuracy, fake_accuracy, feature_matching_loss = train_step(
                image_batch, generator, discriminator, feature_extractor, batch_size, latent_dim, update_discriminator)


            gen_loss_list.append(gen_loss)
            if update_discriminator:
                disc_loss_list.append(disc_loss)
                real_acc_list.append(real_accuracy)
                fake_acc_list.append(fake_accuracy)

            batch_time = (time.perf_counter() - start_time) * 1000

            batch_count += 1

            if update_discriminator:
                print(
                    f'Epoch {epoch + 1}/{epochs}, Batch {batch_count}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}, Feature Loss: {feature_matching_loss:.4f}, Real Acc: {real_accuracy:.2f}, Fake Acc: {fake_accuracy:.2f}, Time: {batch_time:.2f} ms')
            else:
                print(
                    f'Epoch {epoch + 1}/{epochs}, Batch {batch_count}, Gen Loss: {gen_loss:.4f}, ------------------------------------------------------------------------ Time: {batch_time:.2f} ms')

            discriminator_update_counter += 1

            if batch_count >= steps_per_epoch:
                break

        total_epoch_time = (time.perf_counter() - epoch_start) * 1000

        gen_loss_avg = np.mean(gen_loss_list)
        disc_loss_avg = np.mean(disc_loss_list)
        real_acc_avg = np.mean(real_acc_list)
        fake_acc_avg = np.mean(fake_acc_list)

        print(
            f'--- Epoch Summary: {epoch + 1}/{epochs}, Avg Gen Loss: {gen_loss_avg:.4f}, Avg Disc Loss: {disc_loss_avg:.4f}, Avg Real Acc: {real_acc_avg:.2f}, Avg Fake Acc: {fake_acc_avg:.2f}, Total Epoch Time: {total_epoch_time:.2f} ms ---')

        if (epoch + 1) == 1:
            save_imgs(generator, epoch, latent_dim=100)

        if (epoch + 1) % save_interval == 0:
            manager.save()
            print(f'Saving model and generated images at epoch {epoch + 1}')
            save_imgs(generator, epoch, latent_dim=100)
            for real_images in train_dataset.take(1):
                real_images_np = real_images.numpy()
                plot_real_vs_generated(generator, real_images_np, epoch, latent_dim=latent_dim, examples=10)
                break


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
discriminator, feature_extractor = build_discriminator()

generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-5)

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

train_gan(generator, discriminator, feature_extractor, train_dataset, epochs=100, batch_size=batch_size, latent_dim=100,
          steps_per_epoch=train_size // batch_size)
