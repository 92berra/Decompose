import cv2
import os

input_folder = '../output'
output_folder = '../output-resize'

file_list = os.listdir(input_folder)

for file_name in file_list:
    input_path = os.path.join(input_folder, file_name)
    
    img = cv2.imread(input_path)
    
    resized_img = cv2.resize(img, (256, 256))
    
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, resized_img)

print("이미지 변환 완료")
