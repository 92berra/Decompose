#!/usr/bin/env python

import argparse
import math
import os
import random
import glob

import numpy as np
import tensorflow as tf

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'train-tfrecords-output')
DEFAULT_IMAGES_DIR = os.path.join(SCRIPT_PATH, './datasets/images/target-combine')
DEFAULT_NUM_SHARDS_TRAIN = 1
DEFAULT_NUM_SHARDS_TEST = 1

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class TFRecordsConverter(object):
    """Class that handles converting images to TFRecords."""

    def __init__(self, image_dir, output_dir,
                 num_shards_train):

        self.image_dir = image_dir
        self.output_dir = output_dir
        self.num_shards_train = num_shards_train

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get lists of images and labels.
        self.filenames, self.style_labels, self.character_labels = \
            self.process_image_labels(self.image_dir)

        # Counter for total number of images processed.
        self.counter = 0

    def process_image_labels(self, image_dir):
        total_images = glob.glob(os.path.join(image_dir, '*.png'))
        print("total number of images are ", len(total_images))

        images = []
        style_labels = []
        character_labels = []

        for paths in total_images:
            name, _ = os.path.splitext(os.path.basename(paths))
            style_name = name.split('_')[0]
            character_name = name.split('_')[1]
            file = os.path.abspath(paths)
            images.append(file)
            style_labels.append(style_name)
            character_labels.append(character_name)

        shuffled_indices = list(range(len(total_images)))
        random.seed(12121)
        random.shuffle(shuffled_indices)
        filenames = [images[i] for i in shuffled_indices]
        style_labels = [style_labels[i] for i in shuffled_indices]
        character_labels = [character_labels[i] for i in shuffled_indices]
        
        return filenames, style_labels, character_labels

    def write_tfrecords_file(self, output_path, indices):
        """Writes out TFRecords file."""
        with tf.io.TFRecordWriter(output_path) as writer:
            for i in indices:
                filename = self.filenames[i]
                style_label = int(self.style_labels[i])
                character_label = int(self.character_labels[i])
                im_data = tf.io.read_file(filename)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': _bytes_feature(im_data.numpy()),
                    'image/path': _bytes_feature(filename.encode()),
                    'image/style_label':  _int64_feature(style_label),
                    'image/character_label': _int64_feature(character_label)}))
                writer.write(example.SerializeToString())
                self.counter += 1
                if not self.counter % 1000:
                    print('Processed {} images...'.format(self.counter))

    def convert(self):
        num_files_total = len(self.filenames)
        num_files_train = num_files_total

        print('Processing training set TFRecords...')

        files_per_shard = int(math.ceil(num_files_train /
                                        self.num_shards_train))
        start = 0
        for i in range(1, self.num_shards_train):
            shard_path = os.path.join(self.output_dir,
                                      'train-{}.tfrecords'.format(str(i)))
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        file_indices = np.arange(start, num_files_train, dtype=int)
        final_shard_path = os.path.join(self.output_dir,
                                        'train-{}.tfrecords'.format(
                                            str(self.num_shards_train)))
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('\nProcessed {} total images...'.format(self.counter))
        print('Number of training examples: {}'.format(num_files_train))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store TFRecords files.')
    parser.add_argument('--num-shards-train', type=int,
                        dest='num_shards_train',
                        default=DEFAULT_NUM_SHARDS_TRAIN,
                        help='Number of shards to divide training set '
                             'TFRecords into.')
    parser.add_argument('--image-dir', type=str, dest='image_dir',
                        default=DEFAULT_IMAGES_DIR,
                        help='Directory of combine src and tgt images.')
    args = parser.parse_args()
    converter = TFRecordsConverter(args.image_dir,
                                   args.output_dir,
                                   args.num_shards_train)
    converter.convert()
