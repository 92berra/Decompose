import argparse, glob, os, cv2, numpy as np
from jamo import h2j, j2hcj

DEFAULT_IMAGE_DIR = './datasets/images/test'
DEFAULT_OUTPUT_DIR = './datasets/images/test-split'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def separate_test(img_dir, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir))
        
    images_list = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    jamo_dict = {}

    vowel_1 = ['ㅏ','ㅑ','ㅓ','ㅕ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ']
    vowel_2 = ['ㅗ','ㅛ','ㅜ','ㅠ','ㅡ']
    vowel_3 = ['ㅘ','ㅙ','ㅚ','ㅝ','ㅞ','ㅟ','ㅢ']
    
    for j in range(len(images_list)):
        filename = os.path.basename(images_list[j])
        filename = os.path.splitext(filename)[0]
        split_filename = filename.split('_')

        fontname = split_filename[0]
        char_unicode = split_filename[1]

        sylla = chr(int(char_unicode, 16))
        jamo = j2hcj(h2j(sylla))
        jamo_list = list(jamo)
        jamo_dict[j] = jamo_list

        image = cv2.imread(images_list[j])
        image_copy = image.copy()
        img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        center_points = {}
        center_points_dict = []
        sorted_center_points = []    
    
        # combination type 1 (세,하)
        if jamo_dict[j][1] in vowel_1 and len(jamo_dict[j]) == 2: 

            # len(contours) 갯수만큼 반복
            for i in range(1, len(contours)):

                # contours[i]를 감싼 bbox 생성
                x,y,w,h = cv2.boundingRect(contours[i])
            
                # contours[i] 의 중앙값 계산
                # dictionary 형태로 저장 > key는 i, value는 contours의 중앙값
                center_x = x + w // 2
                center_y = y + h // 2
                center_points[i] = (center_x, center_y)
                center_points_dict.append((i, center_points[i]))

            # x값 기준으로 오름차순 정렬
            sorted_center_points = sorted(center_points_dict, key=lambda x: x[1][0])
        
            # x값 기준으로 오름차순 정렬한 인덱스 가져오기
            sorted_contours_indices = [index for index, _ in sorted_center_points]

            if jamo_dict[j][0] == 'ㅅ':
                initial_component_1 = contours[sorted_contours_indices[0]]
                initial_component_2 = contours[sorted_contours_indices[1]]
                middle_component_1 = contours[sorted_contours_indices[2]]
                middle_component_2 = contours[sorted_contours_indices[3]]

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_1.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_2.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

            else:
                initial_component_1 = contours[sorted_contours_indices[0]]
                initial_component_2 = contours[sorted_contours_indices[1]]
                middle_component = contours[sorted_contours_indices[2]]

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_1.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_2.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

        # combination type 2 (루,요)
        elif jamo_dict[j][1] in vowel_2 and len(jamo_dict[j]) == 2:

            # len(contours) 갯수만큼 반복
            for i in range(1, len(contours)):

                # contours[i]를 감싼 bbox 생성
                x,y,w,h = cv2.boundingRect(contours[i])

                # contours[i] 의 중앙값 계산
                # dictionary 형태로 저장 > key는 i, value는 contours의 중앙값
                center_x = x + w // 2
                center_y = y + h // 2
                center_points[i] = (center_x, center_y)
                sorted_center_points.append((i, center_points[i]))
            
            # y값 기준으로 오름차순 정렬
            sorted_center_points = sorted(sorted_center_points, key=lambda x: x[1][1])
        
            # y값 기준으로 오름차순 정렬한 인덱스 가져오기
            sorted_contours_indices = [index for index, _ in sorted_center_points]
            
            middle_component = contours[sorted_contours_indices[-1]]
        
            mask = np.zeros((256,256,3), np.uint8) + 255
            cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
            file_string = f'{fontname}_{char_unicode}_2.png'
            file_path = os.path.join(output_dir, file_string)
            cv2.imwrite(file_path,mask)
        
            cv2.fillPoly(image_copy,[np.array(middle_component)],(255,255,255))
            file_string = f'{fontname}_{char_unicode}_1.png'
            file_path = os.path.join(output_dir, file_string)
            cv2.imwrite(file_path,image_copy)

        # combination type 3 (되)
        elif jamo_dict[j][1] in vowel_3 and len(jamo_dict[j]) == 2: 
            for i in range(1, len(contours)):

                # contours[i]를 감싼 bbox 생성
                x,y,w,h = cv2.boundingRect(contours[i])
            
                # contours[i] 의 중앙값 계산
                # dictionary 형태로 저장 > key는 contours의 index, value는 contours의 중앙값
                center_x = x + w // 2
                center_y = y + h // 2
                center_points[i] = (center_x + center_y)
                center_points_dict.append((i, center_points[i]))
                
            # x+y값 기준으로 오름차순 정렬
            sorted_center_points = sorted(center_points_dict, key=lambda x: x[1])
                
            # x+y값 기준으로 오름차순 정렬한 인덱스 가져오기
            sorted_contours_indices = [index for index, _ in sorted_center_points]

            initial_component = contours[sorted_contours_indices[0]]
        
            mask = np.zeros((256,256,3), np.uint8) + 255
            cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
            file_string = f'{fontname}_{char_unicode}_1.png'
            file_path = os.path.join(output_dir, file_string)
            cv2.imwrite(file_path,mask)
        
            cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
            file_string = f'{fontname}_{char_unicode}_2.png'
            file_path = os.path.join(output_dir, file_string)
            cv2.imwrite(file_path,image_copy)

        # combination type 4
        elif jamo_dict[j][1] in vowel_1 and len(jamo_dict[j]) == 3:
            pass

        # combination type 5 (좋,은)
        elif jamo_dict[j][1] in vowel_2 and len(jamo_dict[j]) == 3:
            # len(contours) 갯수만큼 반복
            for i in range(1, len(contours)):
            
                # contours[i]를 감싼 bbox 생성
                x,y,w,h = cv2.boundingRect(contours[i])
            
                # contours[i] 의 중앙값 계산
                # dictionary 형태로 저장 > key는 i, value는 contours의 중앙값
                center_x = x + w // 2
                center_y = y + h // 2
                center_points[i] = (center_x, center_y)
                center_points_dict.append((i, center_points[i]))
                        
            # y좌표 기준으로 오름차순 정렬
            sorted_center_points = sorted(center_points_dict, key=lambda x: x[1][1])
                        
            # y좌표 기준으로 오름차순 정렬한 인덱스 가져오기
            sorted_contours_indices = [index for index, _ in sorted_center_points]

            if jamo_dict[j][0] == 'ㅈ':
                initial_component_1 = contours[sorted_contours_indices[0]]
                initial_component_2 = contours[sorted_contours_indices[1]]
                middle_component = contours[sorted_contours_indices[2]]
                final_component_1 = contours[sorted_contours_indices[3]]
                final_component_2 = contours[sorted_contours_indices[4]]

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_1.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_2.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_3.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

            else:
                initial_component = contours[sorted_contours_indices[0]]
                middle_component = contours[sorted_contours_indices[1]]
                final_component = contours[sorted_contours_indices[2]]

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_1.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_2.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_3.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)

        # combination type 6
        elif jamo_dict[j][1] in vowel_3 and len(jamo_dict[j]) == 3: 
            pass
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-dir', type=str, dest='img_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default=DEFAULT_OUTPUT_DIR)
   
    args = parser.parse_args()

    separate_test(args.img_dir, args.output_dir)