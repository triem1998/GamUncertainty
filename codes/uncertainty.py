
import time
from scipy.stats.distributions import chi2
import numpy as np
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import scipy
import os
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from math import ceil

# packages for MCMC
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS,HMC
# IAE and unmixing codes
import IAE_CNN_TORCH_Oct2023 as cnn 
import torch
from general import divergence, NNPU, NMF_fixed_a


def _get_barycenter(Lambda,amplitude=None,model=None,fname=None,max_channel_list=None):
    """
    Reconstruct a barycenter from Lambda
    Parameters
    ----------
    Lambda: lambda used to reconstruct the barycenter
    amplitude: amplitude of X, if: None -> the vector 1
    model: IAE model
    fname: name of IAE model if model is not provided
    max_channel_list: max channel for each radionuclide
    -------
    Output: X
    
    """
    #from scipy.optimize import minimize

    if model is None:
        model = load_model(fname)
    model.nneg_output=True

    PhiE,_ = model.encode(model.anchorpoints)
    
    
    B = []
    for r in range(model.NLayers):
        B.append(torch.einsum('ik,kjl->ijl',torch.as_tensor(Lambda.astype("float32")), PhiE[model.NLayers-r-1]))
    tmp=model.decode(B)
    if max_channel_list is not None:
        for i in range(len(max_channel_list)):
            tmp[:,max_channel_list[i]:,i]=0

    if  amplitude is not None:
        tmp=torch.einsum('ijk,i -> ijk',tmp,torch.as_tensor(amplitude.astype("float32")))
    return tmp.detach().numpy()



# loss function, neg log likelihood = divergence
def cost_function(a,y,X):
    tmp=X.dot(a)
    cost=np.sum(tmp-y*np.log(tmp)) 
    return cost
# calculate std of a using Fisher matrix when X is known
def std_fisher(X,a):
    """
    Calculate std using the Fisher information matrix
    Parameters
    ----------
    a: estimated a
    X: spectral signatures
    --------------
    """
    std=np.zeros(len(a))
    X_reduced=X[:,a>0] # only active radionuclides
    weight_reduced=a[a>0]
    M,N=np.shape(X_reduced)
    fisher=np.zeros((N,N))
    tmp=X_reduced.dot(weight_reduced)
    for i in range(N):
        for j in range(N):
            fisher[i,j]=np.dot(X_reduced[:,i]*X_reduced[:,j],1/tmp)
    var=np.linalg.inv(fisher)
    std[a>0]=np.sqrt(np.diag((var)))
    return std
# loss function for calculate Fisher matrix when X is unknown
def loss_fct(inp_tensor,y_tensor,model,MVP_tensor,max_channel_list,radio):
    """
    Calculate loss function
    Parameters
    ----------
    inp_tensor: input tensor (a,lambda)
    y_tensor: observed spectrum
    model : IAE pre-trained model
    MVP_tensor: Background tensor
    max_channel_list:  max channel for each radionuclide
    --------------
    """
    a=inp_tensor[:-1]
    Lambda=inp_tensor[-1][None]
    X_ten=_get_barycenter_ver2(Lambda,model=model,max_channel_list=max_channel_list)[:,radio[1:]] # estimated X
    X_ten=torch.cat((MVP_tensor,X_ten),1) # concatenante with Bkg
    return torch.sum(torch.matmul(X_ten,a)-y_tensor*torch.log(torch.matmul(X_ten,a))) # cost function
# function return X from lambda
def _get_barycenter_ver2(Lambda,model,max_channel_list=None):
    """
    Same as _get_barycenter
    Return the tensor
    --------------
    """
    Lambda=torch.cat((Lambda,1-Lambda),0)
    model.nneg_output=True
    PhiE,_ = model.encode(model.anchorpoints)
    B = []
    #Lambda=torch.as_tensor( Lambda[np.newaxis,:].astype("float32"))
    for r in range(model.NLayers):
        B.append(torch.einsum('ik,kjl->ijl',Lambda[None, :], PhiE[model.NLayers-r-1]))
    tmp=model.decode(B)
    if max_channel_list is not None:
        for i in range(len(max_channel_list)):
            tmp[:,max_channel_list[i]:,i]=0
    tmp=torch.einsum('ijk,ik-> ijk',tmp,1/torch.sum(tmp,axis=(1)))
    return tmp.squeeze()
# autograd approximate hessian matrix and calculate std
def get_std_autograd(a_est,lambda_est,y,model,MVP,max_channel_list=None):
    """
    Approximate hessian matrix by autograd and calculate std
    Parameters
    ----------
    a_est: estimated a
    y: mesured spectrum
    lambda_est: estimated lambda
    model: pre-trained IAE models
    MVP: normalized Bkg
    --------------
    """
    #tranform into tensor
    y_tensor=torch.tensor( y.astype("float32"),requires_grad=True)
    a_tensor=torch.tensor( a_est.astype("float32"),requires_grad=True)
    MVP_tensor=torch.tensor( MVP.astype("float32"))
    lamb_tensor=torch.tensor(lambda_est[0:1].astype("float32"),requires_grad=True)
    radio=(a_est>0)
    #input tensor (a,lambda)
    inp_tensor=torch.cat((a_tensor[radio],lamb_tensor),0)
    hess=torch.autograd.functional.hessian(lambda t: loss_fct(t,y_tensor,model,MVP_tensor,max_channel_list,radio), inp_tensor).detach().numpy() 
    var=np.linalg.inv(hess)
    tmp=np.sqrt(np.diag((var))*(np.diag((var))>0))
    # return vector std of a and lambda
    std_tmp=np.zeros(len(a_est)+1)
    if len(tmp)-1==len(a_est[a_est>0]):
        std_tmp[:-1][a_est>0]=np.int_(tmp[:-1]) # std a, only active radios
        std_tmp[-1]=np.round(tmp[-1],4)#
    else: #  estimated a of some radios are too small -> std=0 -> reshape output 
        amin=(np.sort(a_est[a_est>0])[len(a_est[a_est>0])+1-len(tmp)-1]+np.sort(a_est[a_est>0])[len(a_est[a_est>0])+1-len(tmp)])/2 
        # amin= 1/2*(min+ near min) # a>amin -> fit dimension   
        std_tmp[:-1][a_est[1]>amin]=np.int_(tmp[:-1])
        std_tmp[-1]=np.round(tmp[-1],4)   
    return std_tmp

def model_mcmc_Xfixed(mask,data,XRec,MVP_tensor,cond=None,min_counting_radio=50.):
    """
    MCMC for a (mixing weight) when X is known
    Parameters
    ----------
    mask: list of active radionuclides
    data: tensor of observed data (spectrum y)
    XRec: tensor of X
    MVP_tensor: tensor of BKg
    cond: constraints for total counting, mixing weight 
    min_counting_radio: minimum counting for each radionuclide
    --------------
    """
    # min max values for total countings, mixing weight 
    if cond is None:
        minmax_counting_all,minmax_a=[100.0,None],[[0.0]*(len(mask)+1),[1.0]*(len(mask)+1)] ## init
    else:
        minmax_counting_all,minmax_a=cond
        
    # Bkg: [min total counting * min a0, 1.2 *sum(data)]
    coefs_Bkg=pyro.sample('a_Bkg',dist.Uniform(torch.tensor([minmax_counting_all[0]*minmax_a[0][0]])/torch.sum(data),
                                               torch.tensor([1.2])))
    latent_var=MVP_tensor*coefs_Bkg
    for i in range(XRec.shape[1]):
        coefs = pyro.sample(f"a_{i+1}",dist.Uniform(torch.tensor([min_counting_radio])/torch.sum(data), torch.tensor([1.0])))
        latent_var+=coefs*XRec[:,i:i+1]
    latent_var=latent_var.squeeze()*torch.sum(data)
    obs = pyro.sample("obs", dist.Poisson(latent_var), obs=data)
    return obs

def model_mcmc(mask,data,model,MVP_tensor,max_channel_list=None,cond=None,min_counting_radio=50.):
    """
    MCMC for a (mixing weight) when X is unknown
    Parameters
    ----------
    mask: list of active radionuclides
    data: tensor of observed data (spectrum y)
    model: IAE pre-trained model
    MVP_tensor: tensor of BKg
    max_channel_list:  max channel for each radionuclide
    cond: constraints for total counting, mixing weight 
    min_counting_radio: minimum counting for each radionuclide
    --------------
    """
    # min max values for total countings, mixing weight 
    if cond is None:
        minmax_counting_all,minmax_a,minmax_lambda=[100.0,None],[[0.0]*(len(mask)+1),[1.0]*(len(mask)+1)],[0.0,1.0] ## init
    else:
        minmax_counting_all,minmax_a,minmax_lambda=cond
    Lambda=pyro.sample('lambda',dist.Uniform(torch.tensor([minmax_lambda[0]]), torch.tensor([minmax_lambda[1]])))
    Lambda=torch.cat((Lambda,1-Lambda),0)
    # IAE decodes
    PhiE,_ = model.encode(model.anchorpoints)
    B = []
    for r in range(model.NLayers):
        B.append(torch.einsum('ik,kjl->ijl',Lambda[None, :] , PhiE[model.NLayers-r-1]))
    XRec=model.decode(B)    
    if max_channel_list is not None:
        for i in range(len(max_channel_list)):
            XRec[:,max_channel_list[i]:,i]=0
    XRec=torch.einsum('ijk,ik-> ijk',XRec,1/torch.sum(XRec,axis=(1))).squeeze()
    # Bkg: [min counting * min a0, 1.2 *sum(data)]
    coefs_Bkg=pyro.sample('a_Bkg',dist.Uniform(torch.tensor([minmax_counting_all[0]*minmax_a[0][0]])/torch.sum(data),
                                               torch.tensor([1.2])))
    latent_var=MVP_tensor*coefs_Bkg
    for i in range(len(mask)):
        coefs = pyro.sample(f"a_{i+1}",dist.Uniform(torch.tensor([min_counting_radio])/torch.sum(data), torch.tensor([1.0])))
        latent_var+=coefs*XRec[:,mask[i]:mask[i]+1]
    latent_var=latent_var.squeeze()*torch.sum(data)
    obs = pyro.sample("obs", dist.Poisson(latent_var), obs=data)
    return obs

def get_optimal_coverage_interval(data,list_level=np.array([68.2,95.4,99.7])/100):
    """
    Return HPG credible interval 
    Parameters
    ----------
    data: 3D-array: m*n*h, m: number of data (spectrum y), n: number of MCMC samples, h: number of parameters (a,lambda)
    list_level: nominal level
    output: 4D-array: k*m*h*2, k: number of level
    --------------
    """
    sort_data=np.sort(data,1)   
    n=sort_data.shape[1]
    list_coverage=[]
    for k in list_level:
        length_cov_interval=sort_data[:,ceil(n*k)-1:,:]-sort_data[:,0:n-ceil(n*k)+1,:]
        position=np.argmin(length_cov_interval,1)
        tmp=np.zeros((data.shape[0],data.shape[2],2))
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                tmp[i,j,0]=sort_data[i,position[i,j],j]
                tmp[i,j,1]=sort_data[i,position[i,j]+ceil(n*k)-1,j]
        list_coverage+=[tmp]          
    return np.array(list_coverage)
