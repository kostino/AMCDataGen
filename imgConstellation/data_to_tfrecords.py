import numpy as np
import tensorflow as tf
import os
from PIL import Image

# Dataset parameters
# Image Resolution
img_resolution = (224, 224, 3)
# Dataset Paths
data_root = 'dataset5'
data_root_iq = 'dataset5_iq'
data_root_sig_cum = 'dataset5_cum'
# Total images per Modulation Scheme per SNR
img_mod_snr = 15000
# Images per CUDA batch and number of batches
cuda_batch_size = 5
cuda_batches = img_mod_snr // cuda_batch_size
# SNRs
snrs = [0, 5, 10, 15]
snrs.sort()
# Modulation Schemes
mod_schemes = ['16APSK', '16PAM', '16QAM', '4PAM', '64APSK', '64QAM', '8PSK', 'QPSK']
mod_schemes.sort()


# TFRecords basic methods
# Source: https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


# Function to parse a single dataset entry
def parse_ds_entry(iq_samples, image, cumulants, snr, mod_idx):
    # Define the dictionary -- the structure -- of our single example
    data = {
        'snr': _int64_feature(snr),
        'mod': _int64_feature(mod_idx),
        # 'iq_samples': _bytes_feature(serialize_array(iq_samples)),
        'raw_image': _bytes_feature(serialize_array(image)),
        'cumulants': _bytes_feature(serialize_array(cumulants)),
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'snr': tf.io.FixedLenFeature([], tf.int64),
        'mod': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'cumulants': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)

    snr = content['snr']
    mod = content['mod']
    cumulants = content['cumulants']
    raw_image = content['raw_image']

    # get our 'feature'-- our image -- and reshape it appropriately
    img = tf.io.parse_tensor(raw_image, out_type=tf.string)
    cumulants = tf.io.parse_tensor(cumulants, out_type=tf.float64)
    # TODO: KEEP IMG DIMS IN TFRECORDS ENTRY
    # img = tf.reshape(img, shape=img_resolution)
    img = tf.io.decode_png(img, channels=3)
    cumulants = tf.reshape(cumulants, shape=(18, 1))
    return (img, cumulants, mod, snr)


def get_dataset(filenames):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filenames)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset


# Program Entry point
if __name__ == '__main__':
    if 'tfrecords' not in os.listdir():
        os.makedirs("tfrecords")
    # Iterate over Modulation Schemes
    for mod_idx, mod in enumerate(mod_schemes):
        # Iterate over SNRs
        for snr in snrs:
            # For each Modulation-SNR combination open a new tfrecords shard
            current_shard_name = "tfrecords/{}_{}.tfrecords".format(mod, snr)
            writer = tf.io.TFRecordWriter(current_shard_name)

            # Iterate over CUDA batches
            for batch in range(cuda_batches):
                # Iterate over number of examples per CUDA batch
                for example_num in range(cuda_batch_size):
                    # Load image, cumulants and I/Q samples
                    # img = Image.open(f"{data_root}/{mod}/{snr}_db/{batch}_{example_num}.png")
                    img = open(f"{data_root}/{mod}/{snr}_db/{batch}_{example_num}.png", "rb").read()
                    # img = np.array(img, dtype='uint8')
                    cumulants = np.fromfile(f"{data_root_sig_cum}/{mod}/{snr}_db/{batch}_{example_num}.cum", np.complex128)
                    iq_samples = np.fromfile(f"{data_root_iq}/{mod}/{snr}_db/{batch}_{example_num}.iq", np.complex128)

                    # Convert complex IQ samples and cumulants arrays to float arrays: [samples.real, samples.imag]
                    iq_samples_flt = np.concatenate((iq_samples.real, iq_samples.imag))
                    cumulants_flt = np.concatenate((cumulants.real, cumulants.imag))

                    # Create dictionary with Example information
                    example = parse_ds_entry(iq_samples_flt, img, cumulants_flt, snr, mod_idx)

                    # Write Example to TFRecords shard
                    writer.write(example.SerializeToString())

            # Close writer
            writer.close()
            print(f"Finished {mod} - {snr}dB")
