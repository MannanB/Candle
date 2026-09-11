
# 
# DOWNLOAD MNIST HERE:
# https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download
# extract archive, rename to "data" and place it in the mnist folder 
#

import numpy as np 
import struct
from array import array
from os.path  import join

import random
import matplotlib.pyplot as plt
import math

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Expected 2049, got {magic}")
            labels = list(f.read())

        with open(images_filepath, "rb") as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Expected 2051, got {magic}")
            data = list(f.read())

        images = [
            [data[i*rows*cols + r*cols : i*rows*cols + (r+1)*cols] for r in range(rows)]
            for i in range(size)
        ]

        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        


def get_data_loader():
    input_path = './mnist/data'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    return mnist_dataloader

def show_images(images, title_texts):
    cols = 2
    rows = math.ceil(len(images) / cols)

    plt.figure(figsize=(10, 4 * rows))

    for index, (image, title_text) in enumerate(zip(images, title_texts), start=1):
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)

        if title_text:
            plt.title(title_text, fontsize=9)

        plt.axis("off")

    plt.tight_layout(pad=1.0)
    plt.show()

if __name__ == "__main__":
    mnist_dataloader = get_data_loader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    images_2_show = []
    titles_2_show = []
    for i in range(0, 2):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(0, 2):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

    show_images(images_2_show, titles_2_show)

    plt.show()