import os
import glob

label = 'characters/50characters.txt'
images_dir = '../../Selfstudy/pytorch/GANs-tutorial/Lab1-VanillaGAN/data/target/2'

unicode_list = []

with open(label, 'rt', encoding='utf-8') as fr:
    for line in fr:
        s = line.strip()
        unicode_list.append(hex(ord(s))[2:].upper())

unicode_list_sort = sorted(unicode_list)

image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))

for f in image_files:
    filename = os.path.basename(f)
    filename_without_extension = os.path.splitext(filename)[0]
    
    #split_filename = filename_without_extension.split('_')
    #character_name = split_filename[1]
    character_name = filename_without_extension
    
    if character_name in unicode_list_sort:
        character_index = unicode_list_sort.index(character_name) + 1
        #new_filename = f"{split_filename[0]}_{character_index:05d}.png"
        #new_filename = f"{character_name}.png"
        new_filename = f"{character_index}.png"
        new_f = os.path.join(images_dir, new_filename)
        
        os.rename(f, new_f)
        print('{} --> {}'.format(f, new_f))
    else:
        print('No match found for {}.'.format(f))
