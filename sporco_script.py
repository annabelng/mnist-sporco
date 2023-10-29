# convolutional part helps with resonator
# more powerful than without it 

# import statements

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
from tqdm import tqdm
from sporco import plot
import platform
from scipy.io import loadmat
import pylab as pl
from keras.datasets import mnist
import matplotlib.pyplot as plt


def load_images():
    # Load training and test images
    (X, y), (test_X, test_y) = mnist.load_data()

    # Reshape and normalise training and test images
    train_ims = np.reshape(X,(60000,28,28))/255.
    test_ims = np.reshape(test_X, (10000,28,28))/255.
    print("loaded images")
    return train_ims, test_ims

def train_dictionary(train_ims, prev_weights = None, curr_iter = 0):
    # for batch gradient keep updating S randomly???
    # use np.randint to get 100 random indices
    # pass these random ints into S to get the 100 images
    # aka training on 100 random images at a time
    # and then refreshing every loop 


    # initialize empty dictionary 
    if prev_weights:
        D0 = np.load(prev_weights)['d1'].squeeze()
        print(D0.shape)
    else:
        D0 = np.random.randn(12,12,16)

    for i in tqdm(range(20), desc = 'training'):
        print(i)
        S = np.transpose(train_ims[:1000,:,:],(1,2,0))

        cri = cnvrep.CDU_ConvRepIndexing(D0.shape, S)
        wl1 = np.ones((1,)*4 + (D0.shape[2:]), dtype=np.float32)
        wgr = np.zeros((D0.shape[2]), dtype=np.float32)
        wgr[0] = 1.0
        lmbda = 0.25 # 0.1, 1, 10, 1000
        mu = 0   
        optx = cbpdn.ConvBPDNGradReg.Options({'Verbose': False, 'MaxMainIter': 1,
                    'rho': 20.0*lmbda + 0.5, 'AutoRho': {'Period': 10,
                    'AutoScaling': False, 'RsdlRatio': 10.0, 'Scaling': 2.0,
                    'RsdlTarget': 1.0}, 'HighMemSolve': True, 'AuxVarObj': False,
                    'L1Weight': wl1, 'GradWeight': wgr})
        optd = ccmod.ConvCnstrMODOptions({'Verbose': False, 'MaxMainIter': 1,
                    'rho': 5.0*cri.K, 'AutoRho': {'Period': 10, 'AutoScaling': False,
                    'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}}, 
                    method='cns')

        D0n = cnvrep.Pcn(D0, D0.shape, cri.Nv, dimN=2, dimC=0, crp=True,
                        zm=optd['ZeroMean'])

        optd.update({'Y0': cnvrep.zpad(cnvrep.stdformD(D0n, cri.Cd, cri.M), cri.Nv),
                    'U0': np.zeros(cri.shpD + (cri.K,))})
        print("loaded constants")

        # training chunk
        # put in for loop and update S to grab random images outside of for loop 
        # 50k iterations? 
        xstep = cbpdn.ConvBPDNGradReg(D0n, S, lmbda, mu, optx)
        dstep = ccmod.ConvCnstrMOD(None, S, D0.shape, optd, method='cns')

        for j in tqdm(range(2500), desc=f'Iteration {i}', leave=False):
            opt = dictlrn.DictLearn.Options({'Verbose': False, 'MaxMainIter': 1})
            d = dictlrn.DictLearn(xstep, dstep, opt)
            D1 = d.solve()
            D0 = D1.squeeze()

        #opt = dictlrn.DictLearn.Options({'Verbose': False, 'MaxMainIter': 5000})
        #d = dictlrn.DictLearn(xstep, dstep, opt)
        #D1 = d.solve()
        #D0 = D1.squeeze()
        np.savez('dict_constants/d1_' + str(curr_iter + i*2500) + '.npz', d1=D1)
        save_visualization_as_png(D0, D1, 'features/8_features' + str(i*2500))

        print("DictLearn solve time: %.2fs" % d.timer.elapsed('solve'), "\n")
    return D0, S, D1

def batch_train_dictionary(train_ims):
    # for batch gradient keep updating S randomly???
    # use np.randint to get 100 random indices
    # pass these random ints into S to get the 100 images
    # aka training on 100 random images at a time
    # and then refreshing every loop 

    # initialize empty dictionary and other training constants
    D0 = np.random.randn(12,12,56)
    wl1 = np.ones((1,)*4 + (D0.shape[2:]), dtype=np.float32)
    wgr = np.zeros((D0.shape[2]), dtype=np.float32)
    wgr[0] = 1.0
    lmbda = 0.25 # 0.1, 1, 10, 1000
    mu = 0 

    for i in range(50000):
        # use np.randint to get 100 random indices
        random_indices = np.random.randint(0, 10000, size=100)
        # access images at each random index and pass into S
        train_slice = [train_ims[rand_index] for rand_index in random_indices]
        S = np.transpose(train_slice,(1,2,0))

        cri = cnvrep.CDU_ConvRepIndexing(D0.shape, S)  
        optx = cbpdn.ConvBPDNGradReg.Options({'Verbose': False, 'MaxMainIter': 1,
                    'rho': 20.0*lmbda + 0.5, 'AutoRho': {'Period': 10,
                    'AutoScaling': False, 'RsdlRatio': 10.0, 'Scaling': 2.0,
                    'RsdlTarget': 1.0}, 'HighMemSolve': True, 'AuxVarObj': False,
                    'L1Weight': wl1, 'GradWeight': wgr})
        optd = ccmod.ConvCnstrMODOptions({'Verbose': False, 'MaxMainIter': 1,
                    'rho': 5.0*cri.K, 'AutoRho': {'Period': 10, 'AutoScaling': False,
                    'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}}, 
                    method='cns')

        D0n = cnvrep.Pcn(D0, D0.shape, cri.Nv, dimN=2, dimC=0, crp=True,
                        zm=optd['ZeroMean'])

        optd.update({'Y0': cnvrep.zpad(cnvrep.stdformD(D0n, cri.Cd, cri.M), cri.Nv),
                    'U0': np.zeros(cri.shpD + (cri.K,))})
        print(i)

        # training chunk
        # put in for loop and update S to grab random images outside of for loop 
        # 50k iterations? 
        xstep = cbpdn.ConvBPDNGradReg(D0n, S, lmbda, mu, optx)
        dstep = ccmod.ConvCnstrMOD(None, S, D0.shape, optd, method='cns')

        # change training loop to just include S and opt?? 
        # 8, 16, 32
        for i in range(10):
            opt = dictlrn.DictLearn.Options({'Verbose': False, 'MaxMainIter': 50000})
            d = dictlrn.DictLearn(xstep, dstep, opt)
            D1 = d.solve()
            D0 = D1.squeeze()
            # save
        if i % 500 == 0:
            np.savez('d1_' + str(i) + '.npz', d1=D1)
        save_visualization_as_png(D0, D1, str(i))

    print("DictLearn solve time: %.2fs" % d.timer.elapsed('solve'), "\n")
    return D0, S, D1

# also add checkpoints
# every 500 iterations get plot and save weights

def save_visualization_as_png(D0, D1, output_path):
    # save plots
    D1 = D1.squeeze()
    fig = plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plot.imview(util.tiledict(D0), title='D0', fig=fig)
    
    plt.subplot(1, 2, 2)
    plot.imview(util.tiledict(D1), title='D1', fig=fig)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    train_ims, test_ims = load_images()
    D0, S, D1 = train_dictionary(train_ims, 'dict_constants/d1_34000.npz',34000)
    save_visualization_as_png(D0, D1, "test_script.png")

if __name__ == "__main__":
    main()
