from __future__ import division
from __future__ import print_function
from builtins import input
import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
import res_utils as ru
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
if not platform.system() == 'Windows':
    X, y = loadlocal_mnist(
            images_path='train-images-idx3-ubyte', 
            labels_path='train-labels-idx1-ubyte')
else:
    X, y = loadlocal_mnist(
            images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')
​
train_ims = np.reshape(X,(60000,28,28))/255.
​
dig_idx = [0, 1, 2, 3, 4, 5, 7, 13,15,17]
lmbda = 5e-2
opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 200,
                              'RelStopTol': 5e-3, 'AuxVarObj': False})
b = cbpdn.ConvBPDN(D1_crop, train_ims[dig_idx,:,:], lmbda, opt, dimK=0)
X = b.solve()
​
D1_crop = np.load('convdicts.npz')['D1_crop']
​
def encode_pix_rgb(im, Vt, Ht, Cv):
    N = Vt.shape[0]
    
    image_vec = 0.0 * ru.cvecl(N, 1)
​
    for m in range(im.shape[0]):
        for n in range(im.shape[1]):
            for c in range(im.shape[2]):
                P_vec = Cv[c] * (Vt ** m) * (Ht ** n)
​
                image_vec += P_vec * im[m, n, c]
            
    return image_vec
​
def encode_pix(im, Vt, Ht):
    N = Vt.shape[0]
    
    image_vec = 0.0 * ru.cvecl(N, 1)
​
    for m in range(im.shape[0]):
        for n in range(im.shape[1]):
            P_vec = (Vt ** m) * (Ht ** n)
​
            image_vec += P_vec * im[m, n]
            
    return image_vec
​
def whiten(X,fudge=1e-9):
 
   # the matrix X should be observations-by-components
 
   # get the covariance matrix
    Xcov = np.dot(X.T,X)
 
   # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)
 
   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d+fudge))
    print(D)
​
   # whitening matrix
    W = np.dot(np.dot(V, D), V.T)
 
   # multiply by the whitening matrix
    X_white = np.dot(X, W)
 
    return X_white, W
​
def update_resonator_digit(codebooks,resonator,scene):
    resonator_update = np.ones((len(codebooks),v_size),dtype=complex)
    for i in range(len(codebooks)):
        new_code = scene
        for j in range(len(codebooks)):
            if i != j:
                new_code = new_code*(resonator[j,:]**-1)
        new_code = np.dot(codebooks[i].T,np.dot(np.conj(codebooks[i]),new_code.T))    
#         new_code = np.dot(outer_products[i],new_code.T)
        new_code = new_code / np.abs(new_code)
        resonator_update[i,:] = new_code
    return resonator_update
​
def update_resonator_digit_async(codebooks,resonator,scene):
    resonator_update = copy.copy(resonator)
    for i in range(len(codebooks)):
        new_code = scene
        for j in range(len(codebooks)):
            if i != j:
                new_code = new_code*(resonator_update[j,:]**-1)
        new_code = np.dot(codebooks[i].T,np.dot(np.conj(codebooks[i]),new_code.T))    
        new_code = new_code / np.abs(new_code)
        resonator_update[i,:] = new_code
    return resonator_update
​
def g(x):
    return x / np.abs(x)
​
def gen_res_digit(resonator,codebooks,max_iters,tree):
    res_hist = []
    res_curr = resonator
    for i in range(max_iters):
        res_hist.append(copy.copy(res_curr))
        res_curr = update_resonator_digit_async(codebooks,res_curr,tree)
        if np.mean(np.cos(np.angle(np.ndarray.flatten(res_curr))-np.angle(np.ndarray.flatten(res_hist[-1])))) > 0.99:
            break
    res_hist.append(copy.copy(res_curr))
    return i+1, res_hist
​
def dot_complex(vec1,vec2):
    num = np.dot(np.conj(vec1),vec2)
    denom = np.linalg.norm(vec1)*np.linalg.norm(vec2)
    return np.abs(num)/denom
​
N = int(1e4)
​
# These are special base vectors for position that loop
Vt = ru.cvecl(N, 200)
Ht = ru.cvecl(N, 200)
Ct = ru.cvec(N,14)
​
im_vecs_pixel = np.zeros((10,N),dtype='complex')
​
for i,val in enumerate(dig_idx):
    im_vecs_pixel[i,:] = encode_pix(train_ims[val,:,:],Vt,Ht)
​
im_vecs = np.zeros((10,N),dtype='complex')
for i in range(10):
    # print(X[i,:,:,0,:].shape)
    im_vecs[i,:] = encode_pix_rgb(X[i,:,:,0,:],Vt,Ht,Ct)
​
x = np.reshape(train_ims[dig_idx,:,:],(10,784))
x -= np.mean(x,axis=0)
x_, _ = whiten(x.T)
x_ = x_.T
# matshow(x_ @ x_.T)
# print(x_)
# matshow(np.reshape(x_[1,:],(28,28)))
im_vecs_pixel_white = np.zeros((10,N),dtype='complex')
​
for i,val in enumerate(dig_idx):
    im_vecs_pixel_white[i,:] = encode_pix(np.reshape(x_[i,:],(28,28)),Vt,Ht)
​
x2 = np.reshape(X,(10,28*28*14))
x2 -= np.mean(x2,axis=0)
x2_, _ = whiten(x2.T)
x2_ = x2_.T
matshow(x2_ @ x2_.T)
im_vecs_white = np.zeros((10,N),dtype='complex')
​
for i,val in enumerate(dig_idx):
    im_vecs_white[i,:] = encode_pix_rgb(np.reshape(x2_[i,:],(28,28,14)),Vt,Ht,Ct)
​
space_dim = np.arange(20,301,20)
n_trials = 500
acc = np.zeros((space_dim.size,n_trials,2))
n_total = np.zeros((space_dim.size,n_trials,2))
print(space_dim)
for j in range(space_dim.size):
​
    V_code = np.zeros((space_dim[j],N),dtype='complex')
    H_code = np.zeros((space_dim[j],N),dtype='complex')
    for k in range(space_dim[j]):
        V_code[k,:] = Vt**k
        H_code[k,:] = Ht**k
        
    for k in range(n_trials):
​
        resonator = np.ones((3,N),dtype=complex)
        for i in range(3):
        #     resonator[i,:] = np.sum(codebooks[i],axis=0)
            resonator[i,:] = np.random.normal(0.0,1.0,size=(1,N)) + 1j*np.random.normal(0.0,1.0,size=(1,N))
            resonator[i,:] = resonator[i,:]/ np.abs(resonator[i,:])
        codebooks = [im_vecs,V_code,H_code]
        num_iters = 20
        
        im_seed = np.random.randint(10)
        x_shift = np.random.randint(space_dim[j])
        y_shift = np.random.randint(space_dim[j])
​
        n_iters, res_hist = gen_res_digit(resonator,codebooks,num_iters,im_vecs[im_seed,:]*Vt**y_shift*Ht**x_shift)
​
        progs = []
        for i in range(3):
            c = codebooks[i]
            corrs = np.zeros((n_iters,c.shape[0]),dtype=complex)
            for l in range(n_iters-1,n_iters):
                for m in range(c.shape[0]):
        #             print(k,i,j,codebooks[i].shape)
                    corrs[l,m] = dot_complex(res_hist[l][i,:],codebooks[i][m,:])
        #             corrs[k,j] = np.mean(np.cos(np.abs(np.angle(res_hist[k][i,:] - codebooks[i][j,:]))))
#                     print(corrs[l,m])
            progs.append(corrs)
#         print('.',im_seed,x_shift,y_shift)
#         print(np.argmax(progs[0][-1,:]),np.argmax(progs[1][-1,:]),np.argmax(progs[2][-1,:]))
        if np.argmax(progs[0][-1,:]) == im_seed and np.argmax(progs[1][-1,:]) == y_shift and np.argmax(progs[2][-1,:]) == x_shift:
            acc[j,k,0] = 1
​
            n_total[j,k,0] = n_iters
    print('sparse',space_dim[j],np.sum(acc[j,:,0]),np.sum(n_total[j,:,0]))
​
#     acc = 0
#     n_total = 0
    for k in range(n_trials):
​
        resonator = np.ones((3,N),dtype=complex)
        for i in range(3):
        #     resonator[i,:] = np.sum(codebooks[i],axis=0)
            resonator[i,:] = np.random.normal(0.0,1.0,size=(1,N)) + 1j*np.random.normal(0.0,1.0,size=(1,N))
            resonator[i,:] = resonator[i,:]/ np.abs(resonator[i,:])
        codebooks = [im_vecs_pixel,V_code,H_code]
        num_iters = 20
                
        im_seed = np.random.randint(10)
        x_shift = np.random.randint(space_dim[j])
        y_shift = np.random.randint(space_dim[j])
        
        n_iters, res_hist = gen_res_digit(resonator,codebooks,num_iters,im_vecs_pixel[im_seed,:]*Vt**y_shift*Ht**x_shift)
​
        progs = []
        for i in range(3):
            c = codebooks[i]
            corrs = np.zeros((n_iters,c.shape[0]),dtype=complex)
            for l in range(n_iters-1,n_iters):
                for m in range(c.shape[0]):
        #             print(k,i,j,codebooks[i].shape)
                    corrs[l,m] = dot_complex(res_hist[l][i,:],codebooks[i][m,:])
        #             corrs[k,j] = np.mean(np.cos(np.abs(np.angle(res_hist[k][i,:] - codebooks[i][j,:]))))
#                     print()
            progs.append(corrs)
        if np.argmax(progs[0][-1,:]) == im_seed and np.argmax(progs[1][-1,:]) == y_shift and np.argmax(progs[2][-1,:]) == x_shift:
            acc[j,k,1] = 1
            n_total[j,k,1] = n_iters
    print('pixel',space_dim[j],np.sum(acc[j,:,1]),np.sum(n_total[j,:,1]))
​
fig,(ax1,ax2) = subplots(1,2,figsize=(10,5))
ax1.plot(space_dim,np.mean(acc[:,:,0],axis=1),label='Sparse')
ax1.plot(space_dim,np.mean(acc[:,:,1],axis=1),label='Pixel')
ax1.set_xlabel('Number of x and y positions')
ax1.set_ylabel('Average accuracy')
ax1.legend()
# figure(figsize=(5,5))
ax2.plot(space_dim,np.mean(n_total[:,:,0],axis=1),label='Sparse')
ax2.plot(space_dim,np.mean(n_total[:,:,1],axis=1),label='Pixel')
ax2.set_xlabel('Number of x and y positions')
ax2.set_ylabel('Average number of iterations')
ax2.legend()
​
space_dim = np.arange(20,301,20)
n_trials = 500
acc2 = np.zeros((space_dim.size,n_trials,2))
n_total2 = np.zeros((space_dim.size,n_trials,2))
print(space_dim)
for j in range(space_dim.size):
​
    V_code = np.zeros((space_dim[j],N),dtype='complex')
    H_code = np.zeros((space_dim[j],N),dtype='complex')
    for k in range(space_dim[j]):
        V_code[k,:] = Vt**k
        H_code[k,:] = Ht**k
        
    for k in range(n_trials):
​
        resonator = np.ones((3,N),dtype=complex)
        for i in range(3):
        #     resonator[i,:] = np.sum(codebooks[i],axis=0)
            resonator[i,:] = np.random.normal(0.0,1.0,size=(1,N)) + 1j*np.random.normal(0.0,1.0,size=(1,N))
            resonator[i,:] = resonator[i,:]/ np.abs(resonator[i,:])
        codebooks = [im_vecs_white,V_code,H_code]
        num_iters = 20
        
        im_seed = np.random.randint(10)
        x_shift = np.random.randint(space_dim[j])
        y_shift = np.random.randint(space_dim[j])
​
        n_iters, res_hist = gen_res_digit(resonator,codebooks,num_iters,im_vecs_white[im_seed,:]*Vt**y_shift*Ht**x_shift)
​
        progs = []
        for i in range(3):
            c = codebooks[i]
            corrs = np.zeros((n_iters,c.shape[0]),dtype=complex)
            for l in range(n_iters-1,n_iters):
                for m in range(c.shape[0]):
        #             print(k,i,j,codebooks[i].shape)
                    corrs[l,m] = dot_complex(res_hist[l][i,:],codebooks[i][m,:])
        #             corrs[k,j] = np.mean(np.cos(np.abs(np.angle(res_hist[k][i,:] - codebooks[i][j,:]))))
#                     print(corrs[l,m])
            progs.append(corrs)
#         print('.',im_seed,x_shift,y_shift)
#         print(np.argmax(progs[0][-1,:]),np.argmax(progs[1][-1,:]),np.argmax(progs[2][-1,:]))
        if np.argmax(progs[0][-1,:]) == im_seed and np.argmax(progs[1][-1,:]) == y_shift and np.argmax(progs[2][-1,:]) == x_shift:
            acc[j,k,0] = 1
​
            n_total[j,k,0] = n_iters
    print('sparse',space_dim[j],np.sum(acc[j,:,0]),np.sum(n_total[j,:,0]))
​
    for k in range(n_trials):
​
        resonator = np.ones((3,N),dtype=complex)
        for i in range(3):
        #     resonator[i,:] = np.sum(codebooks[i],axis=0)
            resonator[i,:] = np.random.normal(0.0,1.0,size=(1,N)) + 1j*np.random.normal(0.0,1.0,size=(1,N))
            resonator[i,:] = resonator[i,:]/ np.abs(resonator[i,:])
        codebooks = [im_vecs_pixel_white,V_code,H_code]
        num_iters = 20
                
        im_seed = np.random.randint(10)
        x_shift = np.random.randint(space_dim[j])
        y_shift = np.random.randint(space_dim[j])
        
        n_iters, res_hist = gen_res_digit(resonator,codebooks,num_iters,im_vecs_pixel_white[im_seed,:]*Vt**y_shift*Ht**x_shift)
​
        progs = []
        for i in range(3):
            c = codebooks[i]
            corrs = np.zeros((n_iters,c.shape[0]),dtype=complex)
            for l in range(n_iters-1,n_iters):
                for m in range(c.shape[0]):
        #             print(k,i,j,codebooks[i].shape)
                    corrs[l,m] = dot_complex(res_hist[l][i,:],codebooks[i][m,:])
        #             corrs[k,j] = np.mean(np.cos(np.abs(np.angle(res_hist[k][i,:] - codebooks[i][j,:]))))
#                     print()
            progs.append(corrs)
        if np.argmax(progs[0][-1,:]) == im_seed and np.argmax(progs[1][-1,:]) == y_shift and np.argmax(progs[2][-1,:]) == x_shift:
            acc2[j,k,1] = 1
            n_total2[j,k,1] = n_iters
    print('pixel',space_dim[j],np.sum(acc[j,:,1]),np.sum(n_total[j,:,1]))
​
fig,(ax1,ax2) = subplots(1,2,figsize=(10,5))
ax1.plot(space_dim,np.mean(acc2[:,:,0],axis=1),label='Sparse (whitened)')
ax1.plot(space_dim,np.mean(acc2[:,:,1],axis=1),label='Pixel (whitened)')
ax1.set_xlabel('Number of x and y positions')
ax1.set_ylabel('Average accuracy')
ax1.legend()
# figure(figsize=(5,5))
ax2.plot(space_dim,np.mean(n_total2[:,:,0],axis=1),label='Sparse (whitened)')
ax2.plot(space_dim,np.mean(n_total2[:,:,1],axis=1),label='Pixel (whitened)')
ax2.set_xlabel('Number of x and y positions')
ax2.set_ylabel('Average number of iterations')
ax2.legend()