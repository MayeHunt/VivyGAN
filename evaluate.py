import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model_test4_wgan_image import build_generator, build_discriminator, generate_noise, model_checkpoints, plot_real_vs_generated
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
import pandas as pd
from keras.utils import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt


def save_generated_images(generator, latent_dim=100, num_images=100, output_dir='evaluation/fake'):
    noise = generate_noise(num_images, latent_dim)
    generated_images = generator.predict(noise)
    for i, img in enumerate(generated_images):
        img = np.repeat(img, 3, axis=-1)
        img = Image.fromarray((img * 255).astype('uint8'), 'RGB')
        img.save(f'{output_dir}/image_{i}.png')


def load_generated_images(folder_path, target_size=(299, 299)):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            filepath = os.path.join(folder_path, filename)
            img = load_img(filepath, target_size=target_size, color_mode='rgb')
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            images.append(img)
    images = np.vstack(images)
    return images


def load_and_preprocess_image(image_path, target_size=(299, 299)):
    img = load_img(image_path, target_size=target_size, color_mode='rgb')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def load_images_from_csv(csv_path, set_type, sample_size=100):
    df = pd.read_csv(csv_path)
    df = df[df['set'] == set_type]
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    images = np.vstack([load_and_preprocess_image(fp) for fp in df['file_path']])
    return images


def calculate_fid(model, real_images, generated_images):
    act1 = model.predict(real_images)
    act2 = model.predict(generated_images)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)

    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def display_images(images, title):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        img = (images[i])
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.show()





if __name__ == "__main__":
    # Models initialisation
    generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999)
    discriminator_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999)
    generator = build_generator()
    discriminator = build_discriminator()

    # Loading the checkpoint
    checkpoint_dir = './training_checkpoints_cnn'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    model_checkpoints(checkpoint_dir, generator, discriminator)

    # Generate Images:
    save_generated_images(generator)

    # Calculate FID
    #csv_path = 'dataset_split.csv'
    #generated_path = 'generated/evaluation'
    #real_images = load_images_from_csv(csv_path, set_type='test')
    #generated_images = load_generated_images(generated_path)

    #print(real_images.shape)
    #print(generated_images.shape)
    #display_images(real_images[:9], title='Real Images')
    #display_images(generated_images[:9], title='Generated Images')

    #inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    #fid_score = calculate_fid(inception_model, real_images, generated_images)
    #print("FID Score: ", fid_score)