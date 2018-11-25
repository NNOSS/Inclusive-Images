from array import array
import numpy as np

def getSize(fileobject):
    fileobject.seek(0,2) # move the cursor to the end of the file
    size = fileobject.tell()
    return size

def read_data(list_paths):
    from PIL import Image
    arr = None
    for path in list_paths:
        data = array('B')

        with open(path, 'rb') as f:
            size = getSize(f)
            f.seek(0)
            data.fromfile(f, size)
        EXAMPLE_SIZE = 3* 32*32 + 1
        num = size/EXAMPLE_SIZE
        # print(num)

        examples = [(data[s], data[s+1:s + EXAMPLE_SIZE]) for s in xrange(0, size, EXAMPLE_SIZE)]
        labels, images= zip(*examples)
        images = np.array(images,dtype=np.uint8)
        images = np.reshape(images, (-1,3,32,32))
        images = np.transpose(images, [0,2,3,1])
        # print(images[0])
        images2 = images.astype(np.float32)
        sums = (np.sum(np.power(images2,2),axis=(1,2,3))[:,np.newaxis, np.newaxis, np.newaxis])
        images2 = images2/np.power(sums,.5)
        # print((np.sum(np.power(images2,2),axis=(1,2,3))))
        # print(images2[0])
        if arr is None:
            arr = images
            arr2 = images2
        else:
            arr = np.concatenate([arr, images],0)
            arr2 = np.concatenate([arr2, images2],0)

        # img = Image.fromarray(images[0], 'RGB')
        # img.show()
        # raw_input()
    return arr2, arr

def compare_image(arr, image,ogarr):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()
    image = image.astype(np.float32)
    sums = (np.sum(np.power(image,2)))
    sums = np.power(sums,.5)
    image = image/sums

    # print(image)
    # print(arr)
    arr_weights = arr * image
    similarity = np.sum(arr_weights,(1,2,3))
    # print(similarity[:5])
    indices = np.argsort(similarity)[-5:]
    print(indices)
    print(similarity[indices])
    for index in indices:
        plt.imshow(ogarr[index])
        plt.show()

if __name__ == '__main__':
    FILEPATH = '/Data/OldData/CIFAR10/cifar-10-batches-bin/'
    TRAIN_INPUT = [FILEPATH + 'data_batch_1.bin', FILEPATH + 'data_batch_2.bin',
        FILEPATH + 'data_batch_3.bin', FILEPATH + 'data_batch_4.bin', FILEPATH + 'data_batch_5.bin']
    TRAIN_INPUT = [FILEPATH + 'data_batch_1.bin']
    arr, ogarr = read_data(TRAIN_INPUT)
    compare_image(arr,ogarr[3])
