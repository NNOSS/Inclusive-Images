from array import array
import numpy as np

def getSize(fileobject):
    fileobject.seek(0,2) # move the cursor to the end of the file
    size = fileobject.tell()
    return size

def read_data(list_paths):
    data = array('B')
    from PIL import Image


    for path in list_paths:

        with open(path, 'rb') as f:
            size = getSize(f)
            f.seek(0)
            data.fromfile(f, size)
        EXAMPLE_SIZE = 3* 32*32 + 1
        num = size/EXAMPLE_SIZE

        examples = [(data[s], data[s+1:s + EXAMPLE_SIZE]) for s in xrange(0, num, EXAMPLE_SIZE)]
        labels, images= zip(*examples)
        images = np.array(images,dtype=np.uint8)
        image = np.reshape(images[0], (3,32,32))
        image = np.transpose(image, [1,2,0])
        img = Image.fromarray(image, 'RGB')
        img.show()
        raw_input()

if __name__ == '__main__':
    FILEPATH = '/Data/OldData/CIFAR10/cifar-10-batches-bin/'
    TRAIN_INPUT = [FILEPATH + 'data_batch_1.bin', FILEPATH + 'data_batch_2.bin',
        FILEPATH + 'data_batch_3.bin', FILEPATH + 'data_batch_4.bin', FILEPATH + 'data_batch_5.bin']
    read_data(TRAIN_INPUT)
