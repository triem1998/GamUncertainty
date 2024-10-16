

import pickle
import numpy as np
import time
import sys
import os
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Adam,NAdam,AdamW, SGD, Adagrad, LBFGS,ASGD
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau,ExponentialLR
from pytorch_lightning.callbacks import StochasticWeightAveraging,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import tensorboard 
############################################################
# Main code
############################################################

def _get_optimizer(Optimizer,parameters,learning_rate=1e-4):

    if Optimizer == 0:
        #print("Adam")
        optimizer = Adam(parameters, lr=learning_rate)
    elif Optimizer == 1:
        #print("AdamW")
        optimizer = AdamW(parameters, lr=learning_rate)
    elif Optimizer == 2:
        #print("NAdam")
        optimizer = NAdam(parameters, lr=learning_rate)
    elif Optimizer == 3:
        #print("Adagrad")
        optimizer = Adagrad(parameters, lr=learning_rate, weight_decay=1e-5)
    elif Optimizer == 4:
        #print("SGD")
        optimizer = SGD(parameters, lr=learning_rate)
    elif Optimizer == 5:
        #print("ASGD")
        optimizer = ASGD(parameters, lr=learning_rate)

    return optimizer

def _normalize(X,norm='1',log=False,torchtensor=False):
## normaslise data
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

def _logloss(x1,x2): ## mean of log of mse
    return torch.mean(torch.log(torch.mean((x1-x2)**2,axis=(0,1))))    
    
def _loss(LossOpt='l2'):

    """
    Defines loss functions
    """

    if LossOpt=='l1':
        return torch.nn.L1Loss()
    elif LossOpt=='kl':
        return torch.nn.KLDivLoss()
    elif LossOpt=='log':  
        return _logloss
    
    else:
        return torch.nn.MSELoss()

    ###
    #  ADDING CORRUPTION
    ###

def _corrupt(x,noise_level=None,GaussNoise=True,device="cpu"): # Corrupting the data
    """
    Add noise on data
    x: input data
    noise_level: if int: dB of SNR; float: std of gaussien noise
    GaussNoise: True : same std for each channel (entry) // False: different std for each channnel
    """
    if GaussNoise: # same std for each channel
        if isinstance(noise_level, float):
            noise = noise_level*torch.randn_like(x).to(device)
            return x + noise
        elif isinstance(noise_level, int):
            norm_data = torch.linalg.norm(x,axis=(1))
            noise = torch.randn_like(x).to(device)
            noise_lev=10**(-noise_level/20)*norm_data/torch.linalg.norm(noise,axis=1)
            return x+torch.einsum('ijk,ik->ijk',noise,noise_lev)
    else:
        ## gaussien noise with different std for each channnel
        norm_data = torch.linalg.norm(x,axis=(1))
        noise = torch.randn_like(x).to(device)*x
        noise_lev=10**(-noise_level/20)*norm_data/torch.linalg.norm(noise,axis=1)
        return x+torch.einsum('ijk,ik->ijk',noise,noise_lev)
        
#     else:
#         noise = torch.bernoulli(noise_level*torch.ones_like(x.data)).to(device)
#         return x * noise # We could put additive noise

############################################################
# Saving model from dict
############################################################

def save_model(model,fname='test'):

    params = {"arg_train":model.arg_train,"mean_lambda":model.mean_lambda,"version":model.version,"reg_inv":model.reg_inv,"normalisation":model.normalisation,"anchorpoints":model.anchorpoints, "nsize_fsize":model.nsize_fsize, "nsize_fstride":model.nsize_fstride, "nsize_fnum":model.nsize_fnum,"rho_latcon":model.rho_latcon,"simplex" : model.simplex,"device":model.device,"nonneg_weights":model.nonneg_weights,
              "bounds":model.bounds,"nneg_output":model.nneg_output}

    torch.save({"model":model.state_dict(),"iae_params":params}, fname+".pth")

############################################################
# Loading model from dict
############################################################

def load_model(fname,device="cpu"):

    model_in = torch.load(fname+".pth", map_location=device)
    params = model_in["iae_params"]
    model_state = model_in["model"]

    iae = IAE(input_arg=params,model_load=True)
    iae.load_state_dict(model_state)

    if device=='cuda':
        iae = iae.cuda()

    return iae

############################################################
# IAE args
############################################################

def get_IAE_args(mean_lambda=False,normalisation='inf',anchorpoints=None, nsize_fstride=None, nsize_fnum=None, nsize_fsize=None,
             simplex=False,nonneg_weights=False, rho_latcon=None,reg_inv=1e-6,device="cpu",dropout_rate=None,bounds=None,version="version_December_2022",
                nneg_output=False):
    return {'mean_lambda':mean_lambda,'rho_latcon':rho_latcon,'normalisation':normalisation,'anchorpoints':anchorpoints, 'nsize_fnum':nsize_fnum,'nsize_fstride':nsize_fstride,'nsize_fsize':nsize_fsize, 'reg_inv':reg_inv,'simplex':simplex, 'nonneg_weights':nonneg_weights,'device':device,'dropout_rate':dropout_rate,'version':version,'bounds':bounds,
           'nneg_output':nneg_output}

############################################################
# Main code
############################################################

class IAE(pl.LightningModule):
    """
    Model - input IAE model, overrides other parameters if provided (except the number of layers)
    fname - filename for the IAE model
    anchorpoints - anchor points
    nsize - network structure (e.g. [8, 8, 8, 8] for a 3-layer neural network of size 8)
    active_forward - activation function in the encoder
    active_backward - activation function in the decoder
    res_factor - residual injection factor in the ResNet-like architecture
    reg_parameter - weighting constant to balance between the sample and transformed domains
    cost_weight - weighting constant to balance between the sample and transformed domains in the learning stage
    reg_inv - regularization term in the barycenter computation
    simplex - simplex constraint onto the barycentric coefficients
    nneg_weights - non-negative constraint onto the barycentric coefficients
    nneg_output - non-negative constraint onto the output
    noise_level - noise level in the learning stage as in the denoising autoencoder
    cost_type - cost function (not used)
    optim_learn - optimization algorithm in the learning stage
        (0: Adam, 1: Momentum, 2: RMSprop, 3: AdaGrad, 4: Nesterov, 5: SGD)
    optim_proj - optimization algorithm in the barycentric span projection
    step_size - step size of the optimization algorithms
    niter - number of iterations of the optimization algorithms
    bounds - bounds of latent variable lambda (min, max)
    eps_cvg - convergence tolerance
    verb - verbose mode
    """

    def __init__(self, input_arg=None,arg_train=None,config=None,model_load=False):
        """
        Initialization
        """
        super(IAE,self).__init__()

        if input_arg is None:
            print("Run the get_arg first")

        ###
        self.bounds=input_arg["bounds"]
        self.nneg_output=input_arg['nneg_output']
        self.anchorpoints = torch.as_tensor(input_arg["anchorpoints"])
        self.simplex = input_arg["simplex"]
        self.num_ap = input_arg["anchorpoints"].shape[0]
        #self.device = input_arg["device"]
        self.nonneg_weights = input_arg["nonneg_weights"]
        self.normalisation = input_arg["normalisation"]
        self.version = input_arg["version"]
        self.reg_inv=input_arg["reg_inv"]
        self.normalisation=input_arg["normalisation"]
        self.Lin = self.anchorpoints.shape[1]
        self.PhiE = None
        #self.device=input_arg["device"]
        self.mean_lambda=input_arg["mean_lambda"]

        if model_load:
            self.arg_train=input_arg["arg_train"]
        else:
            self.arg_train=arg_train
        self.lr=self.arg_train["learning_rate"]

        self.LossF = _loss(self.arg_train["LossOpt"])

        if input_arg["rho_latcon"] is None:
            self.rho_latcon = torch.ones((self.NLayers,),device=self.device)
        else:
            self.rho_latcon = input_arg["rho_latcon"]

        ##################### OPTIONS FOR RAY TUNE

        self.nsize_fsize = input_arg["nsize_fsize"]
        self.nsize_fnum = input_arg["nsize_fnum"]
        self.nsize_fstride = input_arg["nsize_fstride"]

        if config is not None:
            if "lr" in config:
                self.lr=config["lr"]
            if "fsizefactor" in config:
                self.nsize_fsize = config["fsizefactor"]*input_arg["nsize_fsize"]
            if "nfilterfactor" in config:
                self.nsize_fnum= config["nfilterfactor"]*input_arg["nsize_fnum"]
            if "rholatconfactor" in config:
                self.rho_latcon = config["rholatconfactor"]*input_arg["rho_latcon"]
        self.NLayers = len(self.nsize_fsize)

        #####################

        dim = []
        dim.append(self.Lin)
        Lin = self.Lin

        for r in range(self.NLayers):

            if r ==0:
                Nch_in = self.anchorpoints.shape[2]
            else:
                Nch_in = self.nsize_fnum[r-1]
            Nch_out = self.nsize_fnum[r]
            kern_size = self.nsize_fsize[r]
            stride = self.nsize_fstride[r]
            Lout = np.int_(np.floor(1+1/stride*(Lin-kern_size)))
            dim.append(Lout)
            Lin = Lout

            
            encoder = []
            encoder.append(torch.nn.Conv1d(Nch_in, Nch_out,kern_size,stride=stride,bias=False))
            encoder.append(torch.nn.BatchNorm1d(Nch_out))
            encoder.append(torch.nn.ELU())  # This could be changed based

            setattr(self,'encoder'+str(r+1),torch.nn.Sequential(*encoder))

            # For the lateral connection

            if r >0 and r < self.NLayers:

                encoder = []
                encoder.append(torch.nn.Conv1d(Nch_in, Nch_out,kern_size,stride=stride,bias=False))
                encoder.append(torch.nn.BatchNorm1d(Nch_out))
                encoder.append(torch.nn.ELU())  # This could be changed based

                setattr(self,'encoder_lat'+str(r),torch.nn.Sequential(*encoder))

        self.dim = dim

        for r in range(1,self.NLayers+1):
            if r == self.NLayers:
                Nch_out = self.anchorpoints.shape[2]
            else:
                Nch_out = self.nsize_fnum[self.NLayers-r-1]
            Nch_in = self.nsize_fnum[self.NLayers-r]
            kern_size = self.nsize_fsize[self.NLayers-r]
            stride = self.nsize_fstride[self.NLayers-r]

            decoder = []
            decoder.append(torch.nn.ConvTranspose1d(Nch_in, Nch_out,kern_size,bias=False))

            if r < self.NLayers+1:
                decoder.append(torch.nn.ELU())  # This could be changed based

            setattr(self,'decoder'+str(r),torch.nn.Sequential(*decoder))

            # For the lateral connection
            if r < self.NLayers:

                decoder = []
                decoder.append(torch.nn.ConvTranspose1d(Nch_in, Nch_out,kern_size,bias=False))
                decoder.append(torch.nn.ELU())  # This could be changed based

                setattr(self,'decoder_lat'+str(r),torch.nn.Sequential(*decoder))

    ###
    #  DISPLAY
    ###

    def display(self,epoch,epoch_time,train_acc,rel_acc,pref="Learning stage - ",niter=None):

        if niter is None:
            niter = self.niter

        percent_time = epoch/(1e-12+niter)
        n_bar = 50
        bar = ' |'
        bar = bar + '█' * int(n_bar * percent_time)
        bar = bar + '-' * int(n_bar * (1-percent_time))
        bar = bar + ' |'
        bar = bar + np.str(int(100 * percent_time))+'%'
        m, s = divmod(np.int_(epoch*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run = ' [{:d}:{:02d}:{:02d}<'.format(h, m, s)
        m, s = divmod(np.int_((niter-epoch)*epoch_time), 60)
        h, m = divmod(m, 60)
        time_run += '{:d}:{:02d}:{:02d}]'.format(h, m, s)

        sys.stdout.write('\033[2K\033[1G')
        if epoch_time > 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +np.str(niter)+ ' -- loss  = {0:e}'.format(np.float(train_acc)) + ' -- validation loss = {0:e}'.format(np.float(rel_acc))+bar+time_run+'-{0:0.4} '.format(epoch_time)+' s/epoch', end="\r")
        if epoch_time < 1:
            print(pref+'epoch {0}'.format(epoch)+'/' +np.str(niter)+ ' -- loss  = {0:e}'.format(np.float(train_acc)) + ' -- validation loss = {0:e}'.format(np.float(rel_acc))+bar+time_run+'-{0:0.4}'.format(1./epoch_time)+' epoch/s', end="\r")
    ###
    #  ENCODE
    ###

    def encode(self, X):

        PhiX_lat = []
        PhiE_lat = []

        PhiX = getattr(self,'encoder'+str(1))(torch.swapaxes(X,1,2))
        PhiE = getattr(self,'encoder'+str(1))(torch.swapaxes(self.anchorpoints.clone(),1,2))

        for r in range(1,self.NLayers):
            #print(PhiX.shape)
            if r < self.NLayers:
                PhiX_lat.append(torch.swapaxes(getattr(self,'encoder_lat'+str(r))(PhiX),1,2))
                PhiE_lat.append(torch.swapaxes(getattr(self,'encoder_lat'+str(r))(PhiE),1,2))

            PhiX = getattr(self,'encoder'+str(r+1))(PhiX)
            PhiE = getattr(self,'encoder'+str(r+1))(PhiE)

        PhiX_lat.append(torch.swapaxes(PhiX,1,2))
        PhiE_lat.append(torch.swapaxes(PhiE,1,2))

        return PhiX_lat,PhiE_lat
    ###
    #  DECODE
    ###
    def decode(self, B):
        # B: barycenter
        Xrec = torch.swapaxes(B[0],1,2)

        for r in range(self.NLayers-1):
            
            Xtemp = Xrec
            Btemp = torch.swapaxes(B[r+1],1,2)
            ## upsample to get the same dimension
            up = torch.nn.Upsample(size=self.dim[self.NLayers-r-1], mode='linear', align_corners=True)
            Xrec = up(getattr(self,'decoder'+str(r+1))(Xtemp) + self.rho_latcon[r]*getattr(self,'decoder_lat'+str(r+1))(Btemp))

        Xrec = getattr(self,'decoder'+str(self.NLayers))(Xrec)
        up = torch.nn.Upsample(size=self.dim[0], mode='linear', align_corners=True)
        Xrec = up(Xrec)
        Xrec=torch.swapaxes(Xrec,1,2)
        if self.nneg_output:
            Xrec=Xrec*(Xrec>0)
        return Xrec
    ###
    #  Interpolate
    ###
    def interpolator(self, PhiX, PhiE):

        L = []
        B = []
        ### lateral connexion -> some values of lambda
        # mean lambda: return the weighted average of lambda
        if self.mean_lambda:

            r=0
            PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[self.NLayers-r-1], PhiE[self.NLayers-r-1])
            iPhiE = torch.linalg.inv(PhiE2 + self.reg_inv*torch.linalg.norm(PhiE2,ord=2)* torch.eye(self.anchorpoints.shape[0],device=self.device))
            Lambda = torch.einsum('ijk,ljk,lm', PhiX[self.NLayers-r-1], PhiE[self.NLayers-r-1], iPhiE)
            sum_val = 1

            for r in range(1,self.NLayers):
                PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[self.NLayers-r-1], PhiE[self.NLayers-r-1])
                iPhiE = torch.linalg.inv(PhiE2 + self.reg_inv*torch.linalg.norm(PhiE2,ord=2)* torch.eye(self.anchorpoints.shape[0],device=self.device))
                Lambda += self.rho_latcon[r-1]*torch.einsum('ijk,ljk,lm', PhiX[self.NLayers-r-1], PhiE[self.NLayers-r-1], iPhiE)
                sum_val += self.rho_latcon[r-1]
            Lambda=Lambda/sum_val

            if self.nonneg_weights:
                mu = torch.max(Lambda,dim=1).values-1.0
                for i in range(2*Lambda.shape[1]) : # or maybe more
                    F = torch.sum(torch.maximum(Lambda,mu.reshape(-1,1)), dim=1) - Lambda.shape[1]*mu-1
                    mu = mu + 1/Lambda.shape[1]*F
                Lambda = torch.maximum(Lambda,mu.reshape(-1,1))-mu.reshape(-1,1)

            elif self.simplex:
                ones = torch.ones_like(Lambda,device=self.device)
                mu = (1 - torch.sum(Lambda,dim=1))/torch.sum(iPhiE)
                Lambda = Lambda +  torch.einsum('ij,i -> ij',torch.einsum('ij,jk -> ik', ones, iPhiE),mu)

            for r in range(self.NLayers):
                L.append(Lambda)
                B.append(torch.einsum('ik,kjl->ijl', Lambda, PhiE[self.NLayers-r-1]))

        else: ## Final lambda is a list of all values of lambda

            for r in range(self.NLayers):
                PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[self.NLayers-r-1], PhiE[self.NLayers-r-1])
                iPhiE = torch.linalg.inv(PhiE2 + self.reg_inv*torch.linalg.norm(PhiE2,ord=2)* torch.eye(self.anchorpoints.shape[0],device=self.device))
                Lambda = torch.einsum('ijk,ljk,lm', PhiX[self.NLayers-r-1], PhiE[self.NLayers-r-1], iPhiE)

                if self.nonneg_weights:
                    mu = torch.max(Lambda,dim=1).values-1.0
                    for i in range(2*Lambda.shape[1]) : # or maybe more
                        F = torch.sum(torch.maximum(Lambda,mu.reshape(-1,1)), dim=1) - Lambda.shape[1]*mu-1
                        mu = mu + 1/Lambda.shape[1]*F
                    Lambda = torch.maximum(Lambda,mu.reshape(-1,1))-mu.reshape(-1,1)

                elif self.simplex:
                    ones = torch.ones_like(Lambda,device=self.device)
                    mu = (1 - torch.sum(Lambda,dim=1))/torch.sum(iPhiE)
                    Lambda = Lambda +  torch.einsum('ij,i -> ij',torch.einsum('ij,jk -> ik', ones, iPhiE),mu)

                L.append(Lambda)
                B.append(torch.einsum('ik,kjl->ijl', Lambda, PhiE[self.NLayers-r-1]))

        return B, L

    def fast_interpolation(self, X, Amplitude=None):
        """
        Use directly the ouput of IAE (decode of interpolation of encoded input sample)
        """
        # Estimating the amplitude

        if Amplitude is None:
            _,Amplitude = _normalize(X,norm=self.normalisation)

        # Encode data
        X = torch.as_tensor(X.astype('float32'))
        PhiX,PhiE = self.encode(torch.einsum('ijk,ik->ijk',X,torch.as_tensor(1./Amplitude.astype('float32'))))

        # Define the barycenter
        B, Lambda = self.interpolator(PhiX,PhiE)

        # Decode the barycenter
        XRec = self.decode(B)

        if X.shape[0]==1:
            XRec = torch.einsum('ijk,ik->ijk',XRec,torch.as_tensor(Amplitude))
        else:
            XRec = torch.einsum('ijk,ik->ijk',XRec,torch.as_tensor(Amplitude))
            #XRec = torch.einsum('ijk,ik->ijk',XRec,torch.as_tensor(Amplitude.squeeze()))

        Output = {"PhiX": PhiX, "PhiE": PhiE, "Barycenter": B, "Lambda": Lambda, "Amplitude": Amplitude, "XRec": XRec}

        return Output

    def get_barycenter(self, Lambda, Amplitude=None):
        """
        Return the IAE estimation of a given lambda
        """
        _,PhiE = self.encode(self.anchorpoints)

        if Amplitude is None:
            Amplitude = torch.ones(Lambda[0].shape[0],self.anchorpoints.shape[2]).to(self.device)

        B = []
        for r in range(self.NLayers):
            B.append(torch.einsum('ik,kjl->ijl', Lambda[r], PhiE[self.NLayers-r-1]))

        # Decode the barycenter
        XRec = torch.einsum('ijk,ik -> ijk',self.decode(B),Amplitude)

        return XRec

    def forward(self, x):
        Z,Ze = self.encode(x)
        B,_ = self.interpolator(Z,Ze)
        return self.decode(B)

    def training_step(self, batch, batch_idx):

        # Corrupting the data

        if self.arg_train["noise_level"] is not None:
            x = _corrupt(batch,self.arg_train["noise_level"],self.arg_train["GaussNoise"],device=self.device)
        else:
            x = batch

        # Applying the IAE

        Z,Ze = self.encode(x)
        B,_ = self.interpolator(Z,Ze)
        x_hat = self.decode(B)

        # Computing the cost

        cost = 0
        for r in range(self.NLayers):
            cost += self.rho_latcon[r]* self.LossF(Z[self.NLayers-r-1],B[r])
            ##############
            
            ###############
        loss = (1+self.arg_train["reg_parameter"])*(self.LossF(x_hat, x) + self.arg_train["reg_parameter"]*cost)

        if self.arg_train["nonneg_output"]:
            loss+= -torch.mean(torch.log(1e-16 + x_hat))
        
        self.log("train_loss", loss, on_step=True)
        self.log("reg_train_loss", cost, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        Z,Ze = self.encode(x)
        B,_ = self.interpolator(Z,Ze)
        x_hat = self.decode(B)
        cost = self.LossF(Z[self.NLayers-1],B[0])
        for r in range(1,self.NLayers):
            cost += self.rho_latcon[r-1]* self.LossF(Z[self.NLayers-r-1],B[r])
        acc = -20*torch.log10(self.LossF(x_hat, x)+1e-16)
        loss = (1+self.arg_train["reg_parameter"])*(acc + self.arg_train["reg_parameter"]*cost)

        if self.arg_train["nonneg_output"]:
            loss+= -torch.mean(torch.log(1e-16 + x_hat))

        self.log("validation_loss", loss)
        self.log("reg_validation_loss", cost)
        self.log("validation_accuracy", acc)

        return {"validation_loss": loss, "validation_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["validation_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["validation_accuracy"] for x in outputs]).mean()
        #print(outputs)
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def configure_optimizers(self):
        optimizer = _get_optimizer(self.arg_train["Optimizer"],self.parameters(),learning_rate=self.lr)
        return optimizer


###############################################################################################################################################
#
# Training
#
###############################################################################################################################################

def get_train_args(fname='IAE_model',SWA_niter=None,nonneg_output=False,verb=True,GaussNoise=True,noise_level=None,reg_parameter=0.1,learning_rate=1e-3 ,  batch_size=64,Optimizer=0,normalisation='1',LossOpt='l2',default_root_dir='./CKPT',max_epochs=5000,accumulate_grad_batches=4,auto_scale_batch_size=False,auto_lr_find=False,enable_checkpointing=True,profiler=None):

    return {"SWA_niter":SWA_niter,"fname":fname,"nonneg_output":nonneg_output,"verb":verb,"GaussNoise":GaussNoise,"noise_level":noise_level,"reg_parameter":reg_parameter,"learning_rate":learning_rate,"batch_size":batch_size,"Optimizer":Optimizer,"normalisation":normalisation,"LossOpt":LossOpt,"default_root_dir":default_root_dir,"max_epochs":max_epochs,"accumulate_grad_batches":accumulate_grad_batches,"auto_scale_batch_size":auto_scale_batch_size,"auto_lr_find":auto_lr_find,"enable_checkpointing":enable_checkpointing,"profiler":profiler}

###############################################################################################################################################
# Trainer
###############################################################################################################################################

def training_lightning(XTrain,arg_IAE=None,arg_train=None,from_model=None,Xvalidation=None,checkmodel=False,checkbest=False):

    """
    CPUData : if true, keeps the data local and only transfer the batches to the GPU
    """
    Xtrain=XTrain.copy()
    if torch.cuda.is_available():
        device = 'cuda'
        kwargs = {}
        acc = "gpu"
        Xpus_per_trial = 1
    else:
        device = 'cpu'
        kwargs = {}
        acc = 'cpu'
        Xpus_per_trial = 1

    print("device USED: ",device)

    if device == 'cuda': # if GPU
        torch.backends.cudnn.benchmark = True

    if arg_train is None:
        arg_train = get_train_args()

    if arg_IAE is None:
        print("Please provide arguments for the IAE model")

    ###
    ### normalisation
    ###
    XTrain = torch.as_tensor(_normalize(XTrain,norm=arg_IAE["normalisation"])[0].astype('float32')).to(device)
    arg_IAE["anchorpoints"] = torch.as_tensor(_normalize(arg_IAE["anchorpoints"],norm=arg_IAE["normalisation"])[0].astype('float32')).to(device)
    data_loader = DataLoader(XTrain, batch_size=arg_train["batch_size"], shuffle=True, **kwargs)

    # Initialize the data loader

    if Xvalidation is not None:
        Xvalidation = torch.as_tensor(_normalize(Xvalidation,norm=arg_IAE["normalisation"])[0].astype('float32')).to(device)
        validation_loader = DataLoader(Xvalidation, batch_size=arg_train["batch_size"], shuffle=False, **kwargs)
    else:
        validation_loader = None

    ###
    ###
    ###

    if from_model is not None:
        IAEmodel = load_model(from_model,device=device)
        # This should be done more systematically
        IAEmodel.nonneg_weights = arg_IAE["nonneg_weights"]
        print(IAEmodel.nonneg_weights)
    else:
        IAEmodel = IAE(input_arg=arg_IAE,arg_train=arg_train)
    IAEmodel = IAEmodel.to(device)

    if arg_train["verb"]:
        print("Training step")

    mycb = [] #[StochasticWeightAveraging(swa_lrs=1e-3,swa_epoch_start=2500)]
    #logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

    trainer = pl.Trainer(callbacks=mycb,default_root_dir=arg_train["default_root_dir"],max_epochs=arg_train["max_epochs"],accumulate_grad_batches=arg_train["accumulate_grad_batches"],auto_scale_batch_size=arg_train["auto_scale_batch_size"],auto_lr_find=arg_train["auto_lr_find"],enable_checkpointing=arg_train["enable_checkpointing"],profiler=arg_train["profiler"],accelerator=acc, devices=Xpus_per_trial,logger=logger,log_every_n_steps =100)
    out_train = trainer.fit(IAEmodel, data_loader,) # Callback pour la validation

    if arg_train["SWA_niter"] is not None:

        mycb = [StochasticWeightAveraging(swa_lrs=1e-3,swa_epoch_start=1)]
        trainer = pl.Trainer(callbacks=mycb,default_root_dir=arg_train["default_root_dir"],max_epochs=arg_train["SWA_niter"],accumulate_grad_batches=arg_train["accumulate_grad_batches"],auto_scale_batch_size=arg_train["auto_scale_batch_size"],auto_lr_find=arg_train["auto_lr_find"],enable_checkpointing=arg_train["enable_checkpointing"],profiler=arg_train["profiler"],accelerator=acc, devices=Xpus_per_trial,logger=logger,log_every_n_steps=100)
    out_train = trainer.fit(IAEmodel, data_loader,) # Callback pour la validation
    
    if checkbest:
        if arg_train["verb"]:
            print("Validation step")
        out_val = trainer.validate(IAEmodel, validation_loader,ckpt_path="best",verbose=True)
    else:
        out_val = None
   
    
    # Saving the model

    if from_model is not None:
        fname_out = from_model+'_restart'
    else:
        fname_out = arg_train["fname"]
    save_model(IAEmodel,fname=fname_out)
    
     ## calculate bound of lambda
    model=load_model(fname_out)
    ########
    res=bsp(Xtrain,model,lr=0.01,optim=1,epochs=100)
    bnds=()
    for i in range(len(np.min(res[1],0))):
        tmp=(np.min(res[1],0)[i],np.max(res[1],0)[i])
        bnds=bnds+(tmp,)
    model.bounds=bnds
    #print(bnds)
    print(model.bounds)
    save_model(model,fname=fname_out)
   
    return IAEmodel, out_train, out_val,trainer

###############################################################################################################################################
# Parameter fitting
###############################################################################################################################################


def bsp(x,model=None,fname=None,a0=None,Lambda0=None,epochs= 100,LossOpt='l2',line_search_fn='strong_wolfe',tol=1e-6,lr=1,tolerance_grad=1e-10,history_size=1000,optim=0):
    """
    Find lambda that the IAE output of this lambda is close to the input x
    x - input (can be noisy data)
    model- IAE model
    fname - if model is not avaible, provide name of IAE model
    a0 - initial amplitude of input 
    Lambda0 - initil value of lambda for searching
    epochs - number of iteration
    LossOpt - Loss function used
    optim - if -1: LBFGS, else: adam, adamw, ... in _get_optimizer
    """

    if model is None:
        model = load_model(fname)
    r=0
    PhiE,_ = model.encode(model.anchorpoints)
    PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[model.NLayers-r-1], PhiE[model.NLayers-r-1])
    iPhiE = torch.linalg.inv(PhiE2 + model.reg_inv*torch.linalg.norm(PhiE2,ord=2)* torch.eye(model.anchorpoints.shape[0],device=model.device))
    d = model.anchorpoints.shape[0]
    b,tx,ty = x.shape

    loss_val = []
    # Initialize a0
    if a0 is None:
        _,a = _normalize(x,norm=model.normalisation)
    else:
        a = a0
    # Initialize Lambda0, if not provided: lambda0 = lambda estimated by fast interpolation
    if Lambda0 is None:
        Lambda = model.fast_interpolation(x,Amplitude=a0)["Lambda"][0].detach().numpy()
    else:
        Lambda = Lambda0
    ## Loss function
    F = _loss(LossOpt=LossOpt)

    x = torch.as_tensor(x.astype("float32"))

    ####
    ## Function returns the estimation of IAE from the parameters P
    ## P: amplitude and lambda
    def Func(P):
        P = P.reshape(b,-1)
        B = []
        Lambda=P[:,ty:]
        
        if model.simplex: # 
            ones = torch.ones_like(Lambda,device=model.device)
            mu = (1 - torch.sum(torch.abs(Lambda),dim=1))/torch.sum(iPhiE)
            #mu = (1 - torch.sum(Lambda,dim=1))/torch.sum(iPhiE)
            Lambda = Lambda +  torch.einsum('ij,i -> ij',torch.einsum('ij,jk -> ik', ones, iPhiE),mu)
        
        for r in range(model.NLayers):
            #B.append(torch.einsum('ik,kjl->ijl',P[:,1::] , PhiE[model.NLayers-r-1]))
            B.append(torch.einsum('ik,kjl->ijl',Lambda , PhiE[model.NLayers-r-1]))
        return torch.einsum('ijk,ik -> ijk',model.decode(B),P[:,0:ty])
    # Initialize P
    P = np.concatenate((a,Lambda),1).reshape(-1,)
    #print(P)
    Params = torch.tensor(P.astype('float32'), requires_grad=True)

    def Loss():
        optimizer.zero_grad()
        rec = Func(Params)
        loss = F(rec, x)
        loss.backward(retain_graph=True)
        return loss
    
    if optim==-1: ## BFGS
        
        #optimizer = torch.optim.LBFGS([Params],max_iter=epochs,line_search_fn=line_search_fn,tolerance_change=tol)
        optimizer =torch.optim.LBFGS([Params],max_iter=epochs,line_search_fn=line_search_fn,tolerance_change=tol,
                                 tolerance_grad=tolerance_grad,lr=lr,history_size=history_size)
        optimizer.step(Loss)
        #print(Params)
        
    else: 
        optimizer=_get_optimizer(optim,[Params],learning_rate=lr)
        for i in range(epochs):
            #print(Params)
            optimizer.step(Loss)
    
    # Params obtained
    Params = Params.reshape(b,-1)
    a,L = Params[:,0:ty].detach().numpy(),Params[:,ty:]
    #L=Params[:,ty:]
    if model.simplex:
        ones = torch.ones_like(L,device=model.device)
        #mu = (1 - torch.sum(L,dim=1))/torch.sum(iPhiE)
        mu = (1 - torch.sum(torch.abs(L),dim=1))/torch.sum(iPhiE)

        L = L +  torch.einsum('ij,i -> ij',torch.einsum('ij,jk -> ik', ones, iPhiE),mu)
    
    
    return _get_barycenter(L.detach().numpy(),amplitude=a,model=model),L.detach().numpy(),a


def bsp2(x,model=None,fname=None,a0=None,Lambda0=None,epochs= 100,LossOpt='log',tol=1e-10,lr=0.01,optim=0):
    """
    Like bsp but more robust (lambda is bounded) and more time-consuming
    Find lambda that the IAE output of this lambda is close to the input x
    x - input (can be noisy data)
    model- IAE model
    fname - if model is not avaible, provide name of IAE model
    a0 - initial amplitude of input 
    Lambda0 - initil value of lambda for searching
    epochs - number of iteration
    LossOpt - Loss function used
    optim - SLSQP
    """

    from scipy.optimize import minimize

    if model is None:
        model = load_model(fname)
    r=0
    PhiE,_ = model.encode(model.anchorpoints)
    PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[model.NLayers-r-1], PhiE[model.NLayers-r-1])
    iPhiE = torch.linalg.inv(PhiE2 + model.reg_inv*torch.linalg.norm(PhiE2,ord=2)* torch.eye(model.anchorpoints.shape[0],device=model.device))
    d = model.anchorpoints.shape[0]
    b,tx,ty = x.shape

    # Initialize Lambda0

    loss_val = []

    if a0 is None:
        _,a = _normalize(x,norm=model.normalisation)
    else:
        a = a0

    if Lambda0 is None:
        Lambda = model.fast_interpolation(x,Amplitude=a0)["Lambda"][0].detach().numpy()

    else:
        Lambda = Lambda0

    F = _loss(LossOpt=LossOpt)
    #### Define contraints
    # simplex: sum of lambda = 1
    def simplex_constraint(param,pos):
        P=param.reshape(b,-1)
        Lamb=P[:,ty:]
        return np.sum(Lamb[pos,:])-1
    #load bound of lambda; bound is calculated in trainning
    bnds=(model.bounds)
    for i in range(ty):
        bnds=((0,None),)+bnds

    constraints=[]
    list_bnds=()
    for i in range(b):
        list_bnds=list_bnds+bnds
        constraints+=[{'type': 'eq','fun':simplex_constraint,'args':(i,)}]
        
    #####
    ## Function returns the estimation of IAE from the parameters P (amplitude and lambda)
    def Func(P):
        #print(P)
        P=torch.tensor(P.astype('float32'))
        P = P.reshape(b,-1)
        B = []
        Lambda=P[:,ty:]
  
        for r in range(model.NLayers):
            #B.append(torch.einsum('ik,kjl->ijl',P[:,1::] , PhiE[model.NLayers-r-1]))
            B.append(torch.einsum('ik,kjl->ijl',Lambda , PhiE[model.NLayers-r-1]))
        return torch.einsum('ijk,ik -> ijk',model.decode(B),P[:,0:ty]).detach().numpy()
        
    P = np.concatenate((a,Lambda),1).reshape(-1,)
    
    ## cost function
    ## arg: input used for estimate lambda
    def get_cost(param,arg):
        #Lamb, Amplitude= param
        
        X=arg[0]
        
        XRec = Func(param)
        if LossOpt=='log':
            return np.mean(np.log(np.sum((X-XRec)**2,axis=(0,1))))
        else:
            return np.sqrt(np.mean((X-XRec)**2))
       
        #return np.sum(Tot-X*np.log(Tot)-X+X*np.log(X))

    sol = minimize(get_cost,x0=P,args=[x],constraints=constraints,  
                       bounds=list_bnds,method='SLSQP',tol=tol,options={'maxiter':epochs,'eps':lr})
    
    #print(sol)
    Params =sol.x
    Params = Params.reshape(b,-1)
    a,L = Params[:,0:ty],Params[:,ty:]    
    XRec=_get_barycenter(L,amplitude=a,model=model)
       
    return _get_barycenter(L,amplitude=a,model=model),L,a
    
    

def bsp_ctr_fast(x,model=None,fname=None,a0=None,Lambda0=None,epochs= 100,LossOpt='log',tol=1e-6,lr=0.01,tolerance_grad=1e-10,history_size=1000,optim=0):
    """
    Find lambda that the IAE output of this lambda is close to the input x
    Use bsp2 for each input (faster than bsp2 for all inputs)
    """
    xrec=[]
    lamb=[]
    Am=[]
    for j in range(x.shape[0]):
        rec=bsp2(x[j:j+1,:,:],model=model,tol=tol,a0=a0,    
            epochs=epochs,optim=optim,lr=lr,)
        Am+=[rec[2]]
        lamb+=[rec[1][0]]
        xrec+=[rec[0][0,:,:]] 
    
        #lambda_list+=[rec2['Lambda'][:,0]]
    xrec=np.array(xrec)
    lamb=np.array(lamb)
    #print(xrec.shape)
    return xrec,lamb,Am



def _get_barycenter(Lambda,amplitude=None,model=None,fname=None):

    """
    Return the IAE estimation of a given lambda
    """
    #from scipy.optimize import minimize

    if model is None:
        model = load_model(fname)

    PhiE,_ = model.encode(model.anchorpoints)

    B = []
    for r in range(model.NLayers):
        B.append(torch.einsum('ik,kjl->ijl',torch.as_tensor(Lambda.astype("float32")), PhiE[model.NLayers-r-1]))
    out=model.decode(B)
    #out=out*(out>0)
    if  amplitude is None:
        #return model.decode(B).detach().numpy()
        return out.detach().numpy()
    else:
        #return torch.einsum('ijk,i -> ijk',model.decode(B),torch.as_tensor(amplitude.astype("float32"))).detach().numpy()
        return torch.einsum('ijk,ik -> ijk',out,torch.as_tensor(amplitude.astype("float32"))).detach().numpy()

###############################################################################################################################################
############################################################################################################################################## NMSE



def NMSE_model(data,Models,RN_NAME='Radio',display=False,SNRVal=None,epochs= 100,lr=0.01,optim=0,LossOpt='log',cst=True,noise_diff_std=True,max_channel=None):
    """
    Return NMSE in function of SNR
    data - input data to evaluate 
    RN_NAME - name of figure 
    display - show a figure of NMSE in function of SNR
    SNRVal - list of SNR values
    epochs - number of iterations
    lr - learning rate
    optim - optimiser used
    LossOpt - loss funcion
    cst - constraint on lambda
    noise_diff_std - same/different std of noise for each channel
    max_channel - max channel
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
   
    plt.rcParams.update({'font.size': 22})
    from matplotlib.ticker import MaxNLocator
    vcol = ['mediumseagreen','crimson','steelblue','darkmagenta','burlywood','khaki','lightblue','darkseagreen','deepskyblue','forestgreen','gold','indianred','midnightblue','olive','orangered','orchid','red','steelblue']
    # nmse function 
    def nmse(x1,x2):
        return -20*np.log10(np.linalg.norm(x1-x2)/np.linalg.norm(x2)),-20*np.log10(np.linalg.norm(x1-x2,axis=(1))/np.linalg.norm(x2,axis=(1)))
    # SNR values
    SNRVal = np.array([0,5,10,15,20,25,30,35,40])
    all_nmse = []
    nmse_mean=[]
    nmse_std=[]
    norm_data = np.linalg.norm(data,axis=(1))

    for r in range(len(SNRVal)):
        ### create noise
        noise = np.random.randn(*data.shape)
        if noise_diff_std:
            noise=noise*data
        if max_channel is not None:
            for d in range(len(max_channel)):
                noise[:,max_channel[d]:,d]=0
        noise_level=10**(-SNRVal[r]/20)*norm_data/np.linalg.norm(noise,axis=1)
        Xn=data+np.einsum('ijk,ik->ijk',noise,noise_level)
        ## optim
        if cst==True: ## slsqp      
            rec = bsp_ctr_fast(Xn,Models,lr=lr,optim=optim,epochs=epochs,LossOpt=LossOpt)
        else: ## adam, etc
            rec = bsp(Xn,Models,lr=lr,optim=optim,epochs=epochs)
        xrec=rec[0]
        xrec,_=_normalize(xrec,norm='1')
        nmse_tot,nmse_ind = nmse(xrec,data)


        nmse_mean+=[np.mean(nmse_ind,axis=0)] 
        nmse_std+=[np.std(nmse_ind,axis=0)]
        all_nmse+=[nmse_tot] # mean for each SNR value

    nmse_mean=np.array(nmse_mean)
    nmse_std=np.array(nmse_std)
    if len(np.shape(nmse_mean))==1:
        nmse_mean=np.reshape(nmse_mean,(len(nmse_mean),1))
        nmse_std=np.reshape(nmse_std,(len(nmse_std),1))

    if display:
        plt.figure(figsize=(15,10))
        for r in range((nmse_mean.shape[1])):

            plt.plot(SNRVal,nmse_mean[:,r],color=vcol[r], marker='o', linestyle='dashed',linewidth=3, markersize=12,label=RN_NAME[r]+' mean')
            plt.plot(SNRVal,nmse_std[:,r],color=vcol[r], marker='x', linestyle='--',linewidth=2, markersize=12,label=RN_NAME[r]+' std')
        plt.legend()
        plt.xlabel('noise level in dB')
        plt.ylabel('NMSE in dB')

    return all_nmse,nmse_mean,nmse_std

















