
#We should get a glob to get all the image files we want to examine
#We should get a dictionary to map file paths from the glob to indexes
#We should read the text files to generate labels for these images_file
#then we should load the images. When we reach a max value we should write to a binary file
from glob import glob
import os
import numpy as np
from PIL import Image
from random import shuffle
import matplotlib.pyplot as plt

def append_binary_file(file_name, bytes_):
    with open(file_name,"ab") as f:
        f.write(bytes_)


class FromJPEG:
    DEBUG = False
    # Image configuration
    data_dir = './Models/data'
    @staticmethod
    def get_image(image_path, width, height, mode, box = None):
        """
        Read image from image_path
        """
        image = Image.open(image_path)
        image = image.resize([width, height], Image.BILINEAR)
        return np.array(image.convert(mode))
    @staticmethod
    def get_batch(image_files, width, height, box = None, mode='RGB'):
        """
        Get a single image
        """
        # print('get file')
        data_batch = np.array(
            [FromJPEG.get_image(sample_file, width, height, mode, box=box) for sample_file in image_files]).astype(np.uint8)
        # Make sure the images are in 4 dimensions
        if len(data_batch.shape) < 4:
            data_batch = data_batch.reshape(data_batch.shape + (1,))

        return data_batch
    @staticmethod
    def get_batches(batch_size,folder,IMAGE_WIDTH,IMAGE_HEIGHT, box = None):
        """
        Generate batches
        """
        # print('start get_batches')
        current_index = 0
        data_files = glob(folder)
        shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, 3
        #TODO
        labels = None
        # print(shape[0])
        while current_index + batch_size <= shape[0]:
            data_batch = get_batch(
                data_files[current_index:current_index + batch_size],
                *shape[1:3], box=box)
            labels_batch = labels[current_index:current_index + batch_size]
            # print('got files')
            current_index += batch_size
            yield data_batch, labels



if __name__ == '__main__':
    IMAGENET_PATH = '/Data/Imagenet/DogsvCats/train/'
    TRAIN_INPUT_SAVE = '/Data/Imagenet/DogsvCats/train_images_64'
    TRAIN_LABEL_SAVE = '/Data/Imagenet/DogsvCats/train_labels_64'
    HEIGHT, WIDTH = 64, 64
    batch_size= 100
    NUM_CLASSES = 1000
    print('Start')

    NUM_CLASSES_SAVE = 1000
    I_2010 = []
    CLASS_PATHS = glob(os.path.join(IMAGENET_PATH,'*'))
    # CLASS_PATHS = []
    # for CLASS_PATH in ALL_CLASS_PATHS:
    #     number = CLASS_PATH.split('/')[5]
    #     if number not in labels_dict:
    #
    #     if 151 < labels_dict[number] < 294:
    #         CLASS_PATHS.append(CLASS_PATH)
    # exit()
    print('Got class paths')
    for i,CLASS_PATH in enumerate(CLASS_PATHS):
        print(i)
        I_2010 += glob(os.path.join(CLASS_PATH,'*.jpg'))
    print('Got Image Paths')
    shuffle(I_2010)
    files_dict = {v.replace(IMAGENET_PATH, '').split('/')[0]: i for i, v in enumerate(CLASS_PATHS)}
    i = 0
    j=0
    print(files_dict)
    print('Being loop')
    while j != -1:
        if i+batch_size <= len(I_2010):
            j = i + batch_size
        else:
            j = -1
            batch_size = len(I_2010) - i + 1
        try:
            labels = np.full(batch_size, 0, dtype = np.int32)
            for k in range(i,j):
                file_name = I_2010[k].replace(IMAGENET_PATH, '')
                class_name = file_name.split('/')[0]
                labels[k-i] = files_dict[class_name]
            images = FromJPEG.get_batch(I_2010[i:j], WIDTH, HEIGHT)
        except:
            i = j
            continue
        # good_classes = np.where(labels<NUM_CLASSES_SAVE)[0]
        # labels = labels[good_classes]
        # images = images[good_classes]
        # for k in range(100):
        #     print(labels[k])
        #     plt.imshow(images[k])
        #     plt.show()
        # exit()
        if len(labels) != len(images):
            print('MISMATCH-------------------------------')
            i = j
            continue
        append_binary_file(TRAIN_INPUT_SAVE,images.tobytes())
        append_binary_file(TRAIN_LABEL_SAVE,labels.tobytes())
        print('ITERS:' ,i)
        i = j
