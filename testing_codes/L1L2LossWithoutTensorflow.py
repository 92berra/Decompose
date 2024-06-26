# import the necessary packages
# from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math, operator
import tensorflow as tf
import os
import sys
import argparse
import glob
from skimage.metrics import structural_similarity as ssim

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# Font Image path which we will test
#TARGET_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../imgs_for_jrnl2/CKFont2_imgs/CKFont2_full/unseen_17_256_wo_ft/tgt_17/')
# TARGET_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '/media/hdd2/pjk/HGL_COMP_study/Korean_journal/SKFont/output_imgs_SKFont_nocheat_new/')
# GENERATED_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '/media/hdd2/pjk/HGL_COMP_study/Korean_journal/jnal_imgs/KomFont_imgs/test_unseen_256_17_new/61_out/')
#GENERATED_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../imgs_for_jrnl2/CKFont2_imgs/CKFont2_full/unseen_17_256_wo_ft/out_17/')
# /media/hdd2/pjk/JRNL2/imgs_for_jrnl2/CKFont2_imgs/CKFont2_full/test_unseen_256_17_w_ft/out_1
# /media/hdd2/pjk/JRNL2/imgs_for_jrnl2/MX_KR_zi2zi/61_image_256
# pjk/JRNL2/imgs_for_jrnl2/CKFont2_imgs/CKFont2_full/unseen_17_256_wo_ft/out_1


TARGET_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../target')
GENERATED_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../output-resize')


num_of_classes = 256
result = 0.0
sum_of_l1_loss = 0.0
sum_of_l2_loss = 0.0
sum_of_ssim=0.0


def read_image(file):
    """Read an image file and convert it into a 1-D floating point array."""
    file_content = tf.read_file(file)
    image = tf.image.decode_jpeg(file_content, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def calculate_L1_loss(real_imgs, gen_imgs, sum_of_l2_loss=0.0, sum_of_l1_loss=0.0, sum_of_ssim=0.0):
    """Read an real and generated image and print their L1 losses"""

    # Check if real image directory exist
    if not os.path.exists(real_imgs):
        print('Error: Real Images Path %s not found.' % real_imgs)
        sys.exit(1)

    # Check if generated image directory exist
    if not os.path.exists(gen_imgs):
        print('Error: Generated Images Path %s not found.' % gen_imgs)
        sys.exit(1)

    # read real images from folder
    real_font_images = glob.glob(os.path.join(real_imgs, '*.png'))
    # check if the images are jpeg
    if len(real_font_images) == 0:
        real_font_images = glob.glob(os.path.join(real_imgs, '*.jpeg'))

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    if all(get_name(path).isdigit() for path in real_font_images):
        target_font_images = sorted(real_font_images, key=lambda path: int(get_name(path)))
    else:
        target_font_images = sorted(real_font_images)

    # read generated images from folder
    gen_font_images = glob.glob(os.path.join(gen_imgs, '*.png'))
    # check if the images are jpeg
    if len(gen_font_images) == 0:
        gen_font_images = glob.glob(os.path.join(gen_imgs, '*.jpeg'))

    if all(get_name(path).isdigit() for path in gen_font_images):
        generated_font_images = sorted(gen_font_images, key=lambda path: int(get_name(path)))
    else:
        generated_font_images = sorted(gen_font_images)

    total_count = 0
    prev_count = 0
    # with tf.Session() as sess:
    for real_fnt_img in target_font_images:
        if total_count - prev_count > 250:
            prev_count = total_count
            print('{} images L1 loss is computed...'.format(total_count))

        target_img_name = os.path.basename(real_fnt_img)
        total_count += 1
        for generated_fnt_img in generated_font_images:
            # getting name for comparison as we are dealing with two diff directories
            generated_img_name = os.path.basename(generated_fnt_img)
            if target_img_name == generated_img_name: # target and generated img's name must be same
                x_real = cv2.imread(real_fnt_img, 0)
                x_predicted = cv2.imread(generated_fnt_img, 0)

                # getting height and width of an image the multiply both
                h_img, w_img = x_real.shape
                total_img_size = h_img*w_img
                MAE = sum(sum(abs(x_real - x_predicted))) / total_img_size
                sum_of_l1_loss = MAE + sum_of_l1_loss

                MSE = sum(sum(pow((x_real-x_predicted),2))) / total_img_size
                sum_of_l2_loss = MSE + sum_of_l2_loss

                # computing ssim
                ssim_measure = ssim(x_real, x_predicted)
                sum_of_ssim = ssim_measure + sum_of_ssim

    # L1 loss avg
    average_l1_loss = sum_of_l1_loss / num_of_classes
    print("Average of L1 Loss: ", average_l1_loss)

    # L2 loss avg
    average_l2_loss = sum_of_l2_loss / num_of_classes
    print("Average of L2 Loss: ", average_l2_loss)

    # SSIM loss avg
    average_ssim = sum_of_ssim / num_of_classes
    print("Average of SSIM: ", average_ssim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_img_dir', type=str, dest='real_image_dir',
                        default=TARGET_FONTS_IMAGE_DIR,
                        help='Directory of real images.')
    parser.add_argument('--gen_img_dir', type=str, dest='generated_image_dir',
                        default=GENERATED_FONTS_IMAGE_DIR,
                        help='Directory of generated images.')
    args = parser.parse_args()

    # Compute MSE
    calculate_L1_loss(args.real_image_dir, args.generated_image_dir, sum_of_l1_loss, sum_of_l2_loss, sum_of_ssim)


