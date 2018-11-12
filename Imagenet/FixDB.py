from glob import glob
import os
import csv
IMAGENET_PATH = '/Data/Imagenet/DogsvCats/train/'
# LABELS_PATH = '/Data/Imagenet/DogsvCats/labels.csv'
# TRAIN_INPUT_SAVE = '/Data/Imagenet/train_images'
# TRAIN_LABEL_SAVE = '/Data/Imagenet/train_labels'
I_2010 = glob(IMAGENET_PATH + '*.jpg')
# ALL_DIRS = [f.replace('.tar', '') for f in I_2010_tar]
# for directory in ALL_DIRS:
#     print(directory)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
iterations = 0


for file_path in I_2010:

    file_name = file_path.replace(IMAGENET_PATH, '')
    folder = file_name[:3]
    new_file_path = IMAGENET_PATH +folder + '/' + file_name
    # print(file_path)
    # print(new_file_path)
    os.rename(file_path, new_file_path)
    # if not iterations % 10000:
    #     print(iterations)
    #     print('Filepath',file_path)
    #     print('NEW FB', new_file_path)
    # iterations+= 1
