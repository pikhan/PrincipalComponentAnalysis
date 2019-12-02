import numpy as np
import matplotlib.pyplot as plt
import os
import pca
import PIL




def compress_images(DATA, k):
    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, 1, 0)
    UT = (np.transpose(PCS)).real
    Xcomp = (np.matmul(Z, UT)).astype(float)
    if not os.path.exists('Output'):
        os.mkdir('Output')
    #for i in range(895)
        #plt.imsave('Output',Xcomp[i], cmap = 'gray')

def load_data(input_dir):
    DATA = []
    obj = os.scandir(path = os.path.abspath(input_dir))
    for entry in obj:
        DATA.append(plt.imread(entry.path).flatten())
        continue
    return (np.asarray(DATA)).astype(float)

compress_images(load_data('/home/polymathykhan/Documents/Project4/Data/ToCompare'), 1)
