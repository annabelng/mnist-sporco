from __future__ import division
from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.admm import cbpdn
from sporco.admm import ccmod
from sporco.dictlrn import dictlrn
from sporco import cnvrep
from sporco import util
from sporco import signal
from sporco import plot
plot.config_notebook_plotting()

from pylab import *
import copy
from mlxtend.data import loadlocal_mnist
import platform
from scipy.io import loadmat
from keras.datasets import mnist
import res_utils as ru

def load_images():
    # Load training and test images
    (X, y), (test_X, test_y) = mnist.load_data()

    # Reshape and normalise training and test images
    train_ims = np.reshape(X,(60000,28,28))/255.
    test_ims = np.reshape(test_X, (10000,28,28))/255.
    return train_ims, test_ims

def load_weights(weights_path):
    # load weights
    D1_crop = np.load(path)['d1']

    dig_idx = [0, 1, 2, 3, 4, 5, 7, 13,15,17]
    lmbda = 5e-2
    opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                                'RelStopTol': 5e-3, 'AuxVarObj': False})
    b = cbpdn.ConvBPDN(D1_crop, train_ims[dig_idx,:,:], lmbda, opt, dimK=0)
    X = b.solve()
    print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'))
    print("Loaded weights from" + path)
    return D1_crop, b, X

def encode_pix_rgb(im, Vt, Ht, Cv):
    N = Vt.shape[0]
    
    image_vec = 0.0 * ru.cvecl(N, 1)

    for m in range(im.shape[0]):
        for n in range(im.shape[1]):
            for c in range(im.shape[2]):
                P_vec = Cv[c] * (Vt ** m) * (Ht ** n)

                image_vec += P_vec * im[m, n, c]
            
    return image_vec

def encode_pix(im, Vt, Ht):
    N = Vt.shape[0]
    
    image_vec = 0.0 * ru.cvecl(N, 1)

    for m in range(im.shape[0]):
        for n in range(im.shape[1]):
            P_vec = (Vt ** m) * (Ht ** n)

            image_vec += P_vec * im[m, n]
            
    return image_vec


