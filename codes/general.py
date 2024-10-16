import time
import numpy as np
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import scipy
import os
import matplotlib.cm as cm
PATH_DATA = '../data/' # CHANGE THAT
PATH = '../codes/'
sys.path.insert(1,PATH)

import sys
sys.path.insert(1,PATH_DATA)

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
vcol = ['mediumseagreen','crimson','steelblue','darkmagenta','burlywood','khaki','lightblue','darkseagreen','deepskyblue','forestgreen','gold','indianred','midnightblue','olive','orangered','orchid','red','steelblue']

import torch




def divergence (X,y,a):
    """
    Calculate the divergence or the negative log likehood 
    Parameters
    ----------
    y: mesured spectrum
    X : spectral signatures
    a :  mixing coefficient or counting
    
    """
    return np.sum(X.dot(a)-y*np.log(X.dot(a)+1e-16)+y*np.log(y+1e-16)-y)

def NNPU(y,X,a0=None,estimed_aMVP=1,niter_max=1000,tol=10**-6):
    """
    NNPU algorithm: estimate the mixing weights a 
    Parameters
    ----------
    y: mesured spectrum
    X : spectral signatures
    a0 : initial mixing coefficient or counting
    estimed_aMVP: 1 - estimate the natural Bkg counting or 0 - do not 
    niter_max: maximum iteration
    tol: stopping criterion
    
    """
    M,N=np.shape(X)
    if a0 is None:
        a0=np.ones(N)/N
    ak=a0.copy()

    niter=0
    err=1 # initial error, > tol
    while (niter<niter_max) & (err>tol) :

        ak1=ak*(np.dot(np.transpose(X),(y/(X.dot(ak)))))
        niter+=1
        if estimed_aMVP==0:
            ak1[0]=ak[0]
        err=np.mean(np.abs((ak1-ak)/(ak+10**-10)))# relative error
        ak=ak1.copy()

    return ak




# normalize function
def _normalize(X,norm='1',log=False,torchtensor=False):

    if torchtensor:
        if len(X.shape) < 3:
            if log:
                X = np.log10(X)
            if norm == '1':
                A = torch.sum(torch.abs(X),(1))
            if norm == '2':
                A = torch.sqrt(torch.sum(torch.square(X),(1)))
            if norm == 'inf':
                A = torch.max(2*torch.abs(X),dim=1).values
            return np.einsum('ij,i-> ij',X,1/A),A
        else:
            if log:
                Y = np.log10(X)
            if norm == '1':
                A = torch.sum(torch.abs(X),(1))
            if norm == '2':
                A = torch.sqrt(torch.sum(torch.square(X),(1,2)))
            if norm == 'inf':
                A = torch.max(torch.max(2*torch.abs(X),dim= 1),dim=2) # Not quite clean
            return np.einsum('ijk,i-> ijk',X,1/A),A
    else:
        if len(X.shape) < 3:
            if log:
                X = np.log10(X)
            if norm == '1':
                A = np.sum(abs(X),axis=1)
            if norm == '2':
                A = np.sqrt(np.sum(X**2,axis=1))
            if norm == 'inf':
                A = 2*np.max(X,axis=1)
            return np.einsum('ij,i-> ij',X,1/A),A
        else:
            if log:
                Y = np.log10(X)
            if norm == '1':
                A = np.sum(abs(X),axis=(1))
            if norm == '2':
                A = np.sqrt(np.sum(X**2,axis=(1,2)))
            if norm == 'inf':
                A = 2*abs(X).max(axis=(1,2))
            return np.einsum('ijk,ik-> ijk',X,1/A),A


def normalize(X,opt='1'):
    Y = X.T
    if opt == '1':
        Y = Y/np.sum(abs(Y),axis=0)
    if opt == '2':
        Y = Y/np.sqrt(np.sum(Y**2,axis=0))
    if opt == 'inf':
        Y = 0.5*Y/np.max(abs(Y),axis=0)
    if opt == 'glob':
        Y = Y/np.max(np.sum(abs(Y),axis=0)) # global normalisation
    return Y.T


def normalize4(X,opt='1'):
    Y=np.einsum('ijk,ik -> ijk',X,1/np.sum(X,axis=1))
    return Y

# convert list of list into list of array
def list_array(data):
    res_list=[]
    for i in range(len(data[0])):
        tmp=[data[j][i] for j in range(len(data))]
        len_tmp=[len(tmp[j]) for j in range(len(tmp))]
        if min(len_tmp)==max(len_tmp):
            res_list+=[np.array(tmp)]
        else:
            res_list+=[tmp]
    return res_list

def NMF_divergence(y,X0,a0,X=None,tol=1e-6,niter_max=1000,norm='1'):
    """
    NMF  estimate non negative X and a 
    Parameters
    ----------
    y: mesured spectrum
    X0 : initial spectral signatures
    X : input spectral signature, for NMSE computation purposes
    a0 : initial mixing coefficient or counting
    niter_max: maximum iteration
    tol: stopping criterion
    
    """
    err=1
    ite=0
    ak=a0.copy()
    Xk=X0.copy()
    loss=[]
    
    if X is not None:
        NMSE_list=[-10*np.log10((np.sum((X0[:,1:]-X[:,1:])**2,axis=0)/np.sum(X[:,1:]**2,axis=0)))] # NMSE
    else:
        NMSE_list = None
        
    while (ite<niter_max) & (err>tol) :

        ak1=ak*Xk.T.dot(y/np.dot(Xk,ak))/(np.sum(Xk,axis=0))        
        Xk1=Xk*((y/Xk.dot(ak1)).reshape(-1,1))
        Xk1[:,0]=Xk[:,0]
        Xk1=Xk1/np.sum(Xk1,axis=0)
        errA= np.mean(np.linalg.norm(ak1-ak)/np.linalg.norm(ak+1e-10))
        errX= np.mean(np.linalg.norm(Xk1-Xk)/np.linalg.norm(Xk+1e-10))
        err=np.maximum(errA,errX)

        ite+=1
        Xk=Xk1.copy()
        ak=ak1.copy()
        loss+=[divergence(Xk,y,ak)]
        if X is not None:
            NMSE_list.append(-10*np.log10((np.sum((Xk[:,1:]-X[:,1:])**2,axis=0)/np.sum(X[:,1:]**2,axis=0))))

    return Xk,ak,np.array(loss),np.array(NMSE_list)

def NMF_fixed_a(y,X0,ak,tol=1e-6,niter_max=100):
    """
    NMF when a is fixed and known, estimate X
    Parameters
    ----------
    y: mesured spectrum
    X0 : initial spectral signatures
    ak : mixing coefficient or counting
    niter_max: maximum iteration
    tol: stopping criterion
    
    """
    err=1
    ite=0
    Xk=X0.copy()
    while (ite<niter_max) & (err>tol) :
        Xk1=Xk*((y/Xk.dot(ak)).reshape(-1,1))
        Xk1[:,0]=X0[:,0]
        Xk1=Xk1/np.sum(Xk1,axis=0)
        err= np.mean(np.linalg.norm(Xk1-Xk)/np.linalg.norm(Xk+1e-10))
        Xk=Xk1
    return Xk

# sort array 3D with new index
def index_array_3D(arr,ind):
    tmp=np.zeros(arr.shape[1:])
    for i in range(len(ind)):
        tmp[:,i]=arr[ind[i],:,i]
    return tmp


