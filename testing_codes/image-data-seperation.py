#!/usr/bin/env python

import argparse
import glob
import io
import os
import random
import shutil
from skimage import img_as_bool,img_as_uint, img_as_ubyte, io as ioo
import natsort

import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../quantitative/')
DEFAULT_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH,"../images")

# Width and height of the resulting image.
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def separate_generated_images(fonts_image_dir, output_dir):
    """
    Separate Generated Images from the Test Data

    """
    #image_dir = os.path.join(output_dir, 'output_imgs_CKFont3_nocheat_new')
    image_dir = os.path.join(output_dir, 'output_imgs')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    print(fonts_image_dir)
    # Provide extensions of targets and output images
    generated_font_images = glob.glob(os.path.join(fonts_image_dir, '*_*-outputs.png'))
    if len(generated_font_images) == 0:
        generated_font_images = glob.glob(os.path.join(fonts_image_dir, '*_*-outputs.png'))

    if len(generated_font_images) == 0:
        raise Exception("Provided Fonts Image directory contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    if all(get_name(path).isdigit() for path in generated_font_images):
        font_images = sorted(generated_font_images, key=lambda path: int(get_name(path)))
    else:
        font_images = sorted(generated_font_images)
        font_images = natsort.natsorted(font_images, None, reverse=False)

    total_count = 0
    prev_count = 0
    for gen_img in font_images:
        # Split names and labels
        name, typeis = os.path.splitext(os.path.basename(gen_img))
        label1 = name.split('-')[0]
        label2 = name.split('-')[1]

        style = label1.split('_')[0]
        char = label1.split('_')[1]

        img_name = label1 + typeis


        # Print image count roughly every 5000 images.
        if total_count - prev_count > 2000:
            prev_count = total_count
            print('{} generated images copied...'.format(total_count))

        total_count += 1

        file_string = '{}.png'.format(total_count)
        file_path = os.path.join(image_dir, img_name)
        shutil.copy(gen_img, file_path)

    print('Finished copying generated {} images.'.format(total_count))

def separate_real_images(fonts_image_dir, output_dir):
    """
    Separate Real Images from the Test Data

    """
    image_dir = os.path.join(output_dir, 'target-images-256-17')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Provide extensions of targets and output images
    target_font_images = glob.glob(os.path.join(fonts_image_dir, '*_*-tgt_font.png'))
    if len(target_font_images) == 0:
        target_font_images = glob.glob(os.path.join(fonts_image_dir, '*_*-tgt_font.png'))

    if len(target_font_images) == 0:
        raise Exception("Provided Fonts Image directory contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    if all(get_name(path).isdigit() for path in target_font_images):
        font_images = sorted(target_font_images, key=lambda path: int(get_name(path)))
    else:
        font_images = sorted(target_font_images)
        font_images = natsort.natsorted(font_images, None, reverse=False)

    total_count = 0
    total_count = 0
    prev_count = 0
    for gen_img in font_images:
        # Split names and labels
        name, typeis = os.path.splitext(os.path.basename(gen_img))
        label1 = name.split('-')[0]
        label2 = name.split('-')[1]

        style = label1.split('_')[0]
        char = label1.split('_')[1]

        img_name = label1 + typeis


        # Print image count roughly every 5000 images.
        if total_count - prev_count > 2000:
            prev_count = total_count
            print('{} target images copied...'.format(total_count))

        total_count += 1

        file_string = '{}.png'.format(total_count)
        file_path = os.path.join(image_dir, img_name)
        shutil.copy(gen_img, file_path)

    print('Finished copying target {} images.'.format(total_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--font-image-dir', type=str, dest='fonts_image_dir',
                        default=DEFAULT_FONTS_IMAGE_DIR,
                        help='Directory of images to use for extracting skeletons.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    separate_generated_images(args.fonts_image_dir, args.output_dir)
    separate_real_images(args.fonts_image_dir, args.output_dir)