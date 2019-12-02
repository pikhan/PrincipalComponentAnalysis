import numpy as np
import matplotlib.pyplot as plt
import os
import pca
import PIL




def compress_images(DATA, k):
    print('test')

def load_data(input_dir):
    DATA = []
    obj = os.scandir(path = os.path.abspath(input_dir))
    for entry in obj:
        DATA.append(plt.imread(entry.path).flatten())
        continue
    return (np.asarray(DATA)).astype(float)

print((load_data('/home/polymathykhan/Documents/Project4/Data/Train')).shape)

