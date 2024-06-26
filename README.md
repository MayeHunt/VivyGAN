# VivyGAN

This repo contains the steps taken for dataset preprocessing, the model training and testing, and prototypes for the [VivyGAN API](https://github.com/MayeHunt/VivyGAN_API).

## Explanation of Files
The prefix of each file denotes what it is used for.

## Preprocessing

### `preprocess_1_split_dataset.py`

This program splits every track of the MAESTRO-v3.0.0 dataset into segments of 4 bars at 120 bpm and quantises the notes to 16th notes. It also adjusts the start time of a segment if it would fall on a period of silence to reduce sparsity. This increases the number of files from 1276 to 85960.

### `preprocess_2_convert_to_pianoroll.py`

This program converts the segmented midi tracks into pianorolls and outputs them as compressed .npz files. As piano rolls tend to encompass the middle of the possible midi note spectrum, the pitch axis is cut to values between 24 and 96, reducing training complexity and sparsity. The sampling frequency is defined as 48, ensuring the temporal resolution of bars are retained. This results in a segment length of 384 beats, which is enforced for model simplicity.

### `preprocess_3_reduce_sparsity.py`

This program loads the piano rolls from the .npz files and ensures that no timesteps contain only 0s, removing the file from the directory if any are found. This drastic approach ensures that all piano rolls contain continuous sound, helping the model recognise that the intended result should not have gaps of silence. This reduces the number of files in the dataset from 85960 to 39721.

### `preprocess_4_convert_to_png.py`

This program loads the piano rolls from the .npz files and converts them to .png image representations. This facilitates the use of the keras `load_img` and `img_to_array` methods for loading data in the training code, meaning all parts of dataset handling are controlled by tensorflow specific functions. It also ensures that the input is in the same format as the output for easy analysis post-generation and a clearer operational pipeline.

### `preprocess_5_csvTestTrainSplit.py`

The final part of preprocessing, this program generates a .csv file containing the relative file path of every .png piano roll segment and assigns each to a train, validate, or testing set at a corresponding ratio of 70:10:20. This functions as a map for the program to load the dataset from without having to load the entire dataset to assign or find files with a specific set.

## Training

### `train_gan.py`

This is the main file used for training the GAN model.

### `train_variation_gan.py`

This is used to train a variation of the GAN model, the main difference being that the generator is designed to accept a different shaped input equal to the dimensions of the input and output piano rolls, with the intent being to use this model to generate variations on existing piano rolls.

## Testing

### `test_copy_csv_sample_to_destination.py`

This program uses the `dataset_split.csv` file to copy a sample of the dataset to a destination folder for evaluation.

### `test_load_play_results.py`

This program can load either .npz or .png piano rolls and output them as .mid MIDI files and .wav audio files. In particular, this allows generated .png files to be listened to for qualitative analysis and also formed a prototype for later implementation into the API code.

### `test_muspy_evaluate.py`

This program generates and loads generated piano rolls and converts them to a Muspy Music class, allowing the use of several Muspy methods for evaluation of the quality of generated outputs through metrics and visualisation. It also features a prototype for model loading from Azure Blob Storage and generating random samples.

## Prototypes

### `prototype_LoadFromCSV.py`

This prototype was used to test the feasibility of loading a dataset from a .csv file of paths.

### `prototype_evaluate.py`

This prototype contains many early test functions that were implemented into other facets of VivyGAN such as: 
* Formatting data for saving to images `save_generated_images()`.
* Loading piano rolls from images `load_generated_images()`.
* Loading and preprocessing images for evaluation `load_and_process_image()`.
* Loading images from paths in a .csv file `load_images_from_csv()` and displaying them `display_images()`.
* Removing short notes from generated samples at a threshold length `remove_short_notes()`.
* Generating a variation on another sample and immediately saving it `generate_from_input()`.

It also features an approach to loading a saved model from checkpoint files and a function for calculating the Frechet Inception Distance of generated samples against real samples.

### `prototype_generate_variations.py`

This prototype was intended as a test for the API route for generating variations using the variations model. It downloads and loads the model from Azure Blob Storage and then loads an input with different function for handling various input types. It then applies noise to that input to generate variations.

## CSVs

### `dataset_split.csv`

This is the .csv file produced by `preprocess_5_csvTestTrainSplit.py` used for getting files with preassigned testing, validate, and training sets.

### `generated_songs_metrics.csv`

This is the .csv file produced by the `test_muspy_evaluate.py` program. This contains metrics for each generated image to evaluate the quality of outputs.
