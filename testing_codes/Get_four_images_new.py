# imprts related to creating paths
import io
import os
import argparse

# imports related to preprocess from pix2pix
import tfimage as im
import time
import tensorflow as tf
import numpy as np
import threading


# setting global variable for counter
# 카운터를 위한 글로벌 변수 설정
index = 0
total_count = 0 

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

#DEFAULT_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '/home/sslab/Desktop/Ammar/comparison/Debbie_SkelGAN/F2F/trg-image-data/trg-hangul-images/')

#DEFAULT_FFG_1 = os.path.join(SCRIPT_PATH, '/home/sslab/Desktop/Ammar/comparison/Debbie_SkelGAN/F2F/image-data-114-2350/images')
#DEFAULT_FFG_2 = os.path.join(SCRIPT_PATH, '/home/sslab/Desktop/Ammar/comparison/Debbie_SkelGAN/F2F/image-data-256-2350/images')
#DEFAULT_FFG_3 = os.path.join(SCRIPT_PATH, '/home/sslab/Desktop/Ammar/comparison/Debbie_SkelGAN/F2F/image-data-512-2350/images')
# DEFAULT_FFG_4 = os.path.join(SCRIPT_PATH, './FFG_newdataset_tests/FFG_11/5_fonts/generated-image-data')
# DEFAULT_FFG_5 = os.path.join(SCRIPT_PATH, './FFG_newdataset_tests/FFG_12/5_fonts/generated-image-data')

#DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '/home/sslab/Desktop/Ammar/comparison/Debbie_SkelGAN/F2F/combine-images/114-256-512/20_fonts')


DEFAULT_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../quantitative/target')
DEFAULT_FFG_1 = os.path.join(SCRIPT_PATH, '../quantitative/output')
DEFAULT_FFG_2 = os.path.join(SCRIPT_PATH, '../../tested_model-50-pjk/quantitative/output')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../quantitative/result')


# 하나의 한글 이미지 정보를 입력 받아 스켈레톤 이미지를 찾아 전처리한 후 나란히 연결함
def combine(src, src_path):
    if args.b1_dir is None:
        raise Exception("missing b1_dir")
    elif args.b2_dir is None:
        raise Exception("missing b2_dir")
    elif args.b3_dir is None:
        raise Exception("missing b3_dir")
    # elif args.b4_dir is None:
    #     raise Exception("missing b4_dir") 
    # elif args.b5_dir is None:
    #     raise Exception("missing b4_dir") 
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path1 = os.path.join(args.b1_dir, basename + ext)
        if os.path.exists(sibling_path1):
            sibling1 = im.load(sibling_path1)
            break
    else:
        raise Exception("could not find sibling1 image for " + src_path)

    for ext in [".png", ".jpg"]:
        sibling_path2 = os.path.join(args.b2_dir, basename + ext)
        if os.path.exists(sibling_path2):
            sibling2 = im.load(sibling_path2)
            break
    else:
        raise Exception("could not find sibling2 image for " + src_path)

    for ext in [".png", ".jpg"]:
        sibling_path3 = os.path.join(args.b3_dir, basename + ext)
        if os.path.exists(sibling_path3):
            sibling3 = im.load(sibling_path3)
            break
    else:
        raise Exception("could not find sibling3 image for " + src_path)

    # for ext in [".png", ".jpg"]:
    #     sibling_path4 = os.path.join(args.b4_dir, basename + ext)
    #     if os.path.exists(sibling_path4):
    #         sibling4 = im.load(sibling_path4)
    #         break
    # else:
    #     raise Exception("could not find sibling4 image for " + src_path)

    # for ext in [".png", ".jpg"]:
    #     sibling_path5 = os.path.join(args.b5_dir, basename + ext)
    #     if os.path.exists(sibling_path5):
    #         sibling5 = im.load(sibling_path5)
    #         break
    # else:
    #     raise Exception("could not find sibling5 image for " + src_path)

    height, width, _ = src.shape
    if height != sibling1.shape[0] or width != sibling1.shape[1] or height != sibling2.shape[0] or width != sibling2.shape[1]:
        raise Exception("differing sizes")
    
    # convert all images to RGB if necessary
    # 두 이미지가 만일 그레이스케일 이미지라면 RGB로 변환
    if src.shape[2] == 1:
        src = im.grayscale_to_rgb(images=src)

    if sibling1.shape[2] == 1:
        sibling1 = im.grayscale_to_rgb(images=sibling1)

    if sibling2.shape[2] == 1:
        sibling2 = im.grayscale_to_rgb(images=sibling2)

    if sibling3.shape[2] == 1:
        sibling3 = im.grayscale_to_rgb(images=sibling3)

    # if sibling4.shape[2] == 1:
    #     sibling4 = im.grayscale_to_rgb(images=sibling4)

    if src.shape[2] == 4:
        src = src[:,:,:3]
    
    if sibling1.shape[2] == 4:
        sibling1 = sibling1[:,:,:3]

    if sibling2.shape[2] == 4:
        sibling2 = sibling2[:,:,:3]

    if sibling3.shape[2] == 4:
        sibling3 = sibling3[:,:,:3]
    
    # if sibling4.shape[2] == 4:
    #     sibling4 = sibling4[:,:,:3]
    
    # if sibling5.shape[2] == 4:
    #     sibling5 = sibling5[:,:,:3]

    return np.concatenate([sibling1, sibling2, sibling3, src], axis=1)


def process(src_path, dst_path, image_dir):
    global index
    global total_count

    # 입력받은 하나의 한글 이미지 경로를 가지고 해당 한글 이미지를 읽음
    total_count += 1
    src = im.load(src_path)

    # 명령어에서 인자로 입력받은 연산이 combine이라면 combine 수행
    if args.operation == "combine":
        dst = combine(src, src_path)
    else:
        raise Exception("invalid operation")
    # combine을 수행한 결과를 파일로 저장함
    im.save(dst, dst_path)

    # 저장한 combine 결과 이미지를 csv 파일에 레이블을 맵핑하여 저장
    file_string = '{}.png'.format(total_count)


complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0


def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining"
              % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now


def generate_hangul_skeleton_combine_images(output_dir):
    # 출력 디렉토리 유무 확인 후 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_dir = output_dir
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    src_paths = []
    dst_paths = []

    # Check if the directory and images already exist?
    # If yes then skip those images else create the paths list
    # combine 결과 이미지가 이미 존재하는 지 확인
    # 만약 존재한다면 그 이미지들을 스킵하고, 그렇지 않으면 경로 리스트를 생성함
    skipped = 0
    for src_path in sorted(im.find(args.input_dir)):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(image_dir, name + ".png")
        if os.path.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)
    
    print("skipping %d files that already exist" % skipped)

    global total
    total = len(src_paths)
    
    print("processing %d files" % total)

    global start
    start = time.time()

    if args.workers == 1:
        with tf.Session() as sess:
            for src_path, dst_path in zip(src_paths, dst_paths):
                process(src_path, dst_path, image_dir)
                complete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                            default=DEFAULT_FONTS_IMAGE_DIR,
                            help="path to folder containing images")
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                            default=DEFAULT_OUTPUT_DIR,
                            help='Output directory to store generated hangul skeleton images and '
                                 'label CSV file.')
    parser.add_argument("--operation", default='combine', choices=["combine"])
    parser.add_argument("--workers", type=int, default=1, help="number of workers")
    # combine
    parser.add_argument("--b1_dir", type=str, default=DEFAULT_FFG_1,
                        help="path to folder containing B images of white characters for combine operation")
    parser.add_argument("--b2_dir", type=str, default=DEFAULT_FFG_2,
                        help="path to folder containing B images of white characters for combine operation")
    parser.add_argument("--b3_dir", type=str, default=DEFAULT_FFG_3,
                        help="path to folder containing B images of white characters for combine operation")
    # parser.add_argument("--b4_dir", type=str, default=DEFAULT_FFG_4,
    #                     help="path to folder containing B images of white characters for combine operation")
    # parser.add_argument("--b5_dir", type=str, default=DEFAULT_FFG_5,
    #                     help="path to folder containing B images of white characters for combine operation")
    args = parser.parse_args()

    generate_hangul_skeleton_combine_images(args.output_dir)
