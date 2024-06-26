
import cv2
import tensorflow as tf
import os
import sys
import argparse
import glob
import io

from skimage.metrics import structural_similarity as ssim

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# Font Image path which we will test
# TARGET_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../FSE-2-F/s2/target-image-data')
# GENERATED_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../FSE-2-F/s2/generated-image-data')

#TARGET_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '/media/hdd2/pjk/HGL_COMP_study/Korean_journal/KomFONT/tgt_imgs_Komfont')
#GENERATED_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '/media/hdd2/pjk/HGL_COMP_study/Korean_journal/KomFONT/output_imgs_Komfont')
#DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '/media/hdd2/pjk/HGL_COMP_study/Korean_journal/KomFONT/SSIM_KomFont')

TARGET_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../target')
GENERATED_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../output-resize')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../quantitative/ssim')

total_num_imgs = 7
result = 0.0
sum_of_ssim = 0.0
sum_of_l2_loss = 0.0


def read_image(file):
    """Read an image file and convert it into a 1-D floating point array."""
    file_content = tf.read_file(file)
    image = tf.image.decode_jpeg(file_content, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def calculate_SSIM(real_imgs, gen_imgs, output_dir, sum_of_ssim=0.0):
    """Read an real and generated image and print their L1 losses"""
    
    output_image_dir = output_dir
    if not os.path.exists(output_image_dir):
        os.makedirs(os.path.join(output_image_dir))

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

    # File to write l1 losses
    l1_loss_csv = io.open(os.path.join(output_dir, 'SSIMStyle1.csv'), 'w', encoding='utf-8')

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
        gen_font_images = glob.glob(os.path.join(gen_imgs, '*.jpg'))

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
            print('{} images ssim is computed...'.format(total_count))

        target_img_name = os.path.basename(real_fnt_img)
        total_count += 1
        for generated_fnt_img in generated_font_images:
            # getting name for comparison as we are dealing with two diff directories
            generated_img_name = os.path.basename(generated_fnt_img)
            if target_img_name == generated_img_name:
                x_real = cv2.imread(real_fnt_img, 0)
                x_predicted = cv2.imread(generated_fnt_img, 0)

                # computing ssim
                ssim_measure = ssim(x_real, x_predicted)
                print("SSIM: %.2f" % ssim_measure)

                # writing l1 loss
                l1_loss_csv.write(u'{},{} \n'.format(generated_fnt_img, ssim_measure))

                if ssim_measure < 0.8 or ssim_measure < 0.6:
                    print("SSIM: %.2f" % ssim_measure)

                sum_of_ssim = ssim_measure + sum_of_ssim

    # SSIM sum
    average_ssim = sum_of_ssim / total_num_imgs
    print("Average of SSIM: ", average_ssim)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_img_dir', type=str, dest='real_image_dir',
                        default=TARGET_FONTS_IMAGE_DIR,
                        help='Directory of real images.')
    parser.add_argument('--gen_img_dir', type=str, dest='generated_image_dir',
                        default=GENERATED_FONTS_IMAGE_DIR,
                        help='Directory of generated images.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to ssim')
    args = parser.parse_args()

    # Compute SSIM
    calculate_SSIM(args.real_image_dir, args.generated_image_dir, args.output_dir, sum_of_ssim)


