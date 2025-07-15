import numpy as np 
import pandas as pd
from torch import tensor,float32,save
import os
import matplotlib.pyplot as plt
import torch
from torch.func import vmap,jacrev

def load_df(name,tag):
    parent_dir = os.path.dirname(os.getcwd())

    density_profiles_dir = os.path.join(parent_dir, "Data_generation/Density_profiles")
    name = os.path.join(parent_dir ,tag, name)
    print("Reading file: ", name)
    df = pd.read_csv(name,delimiter = " ")

    extra=[]
    try:
        nperiod = (list(df.columns)[4])
        df.drop(columns=[nperiod],inplace=True)
        nperiod = int(nperiod)
        extra.append(nperiod)
    except:
        print("No nperiod")
    try:
        mu = (list(df.columns)[4])
        df.drop(columns=[mu],inplace=True) 
        mu = float(mu)
        extra.append(mu)
    except:
        print("No mu")
    try:
        packing = (list(df.columns)[4])
        df.drop(columns=[packing],inplace=True)
        packing = float(packing)
        extra.append(packing)
    except:
        print("No packing")
    return df, extra


def cut_density_windows(df, window_dim,n_windows,window_stack,center_values):
    rhomatrix = df.pivot(index='y', columns='x', values='rho').values
    mulocmatrix = df.pivot(index='y', columns='x', values='muloc').values
    for i in range(n_windows):
        for j in range(n_windows):
            window = rhomatrix[i:i+window_dim,j:j+window_dim]
            center_x_index = int(i+0.5*window_dim)
            center_y_index = int(j+0.5*window_dim)
            center = mulocmatrix[center_x_index,center_y_index]
            window_stack.append(window)
            center_values.append(center)
            
    return window_stack,center_values

def cut_density_windows_torch_unpadded(df, window_dim,n_windows):
    rhomatrix = df.pivot(index='y', columns='x', values='rho').values
    rhotensor = torch.tensor(rhomatrix, dtype=torch.float32)
    unfolded_rho = rhotensor.unfold(0, window_dim, window_dim).unfold(1, window_dim, window_dim)
    windows = unfolded_rho.contiguous().view(-1, window_dim, window_dim)
    return windows


## NOTE: This function IS NOT WORKING CORRECTLY, stride optimized for GPUmem on local
def cut_density_windows_torch_padded(df, window_dim):
    stride = 7
    rhomatrix = df.pivot(index='y', columns='x', values='rho').values
    mulocmatrix = df.pivot(index='y', columns='x', values='muloc').values
    pad_size = window_dim//2
    rhotensor = torch.tensor(rhomatrix, dtype=torch.float32)
    rhotensor = rhotensor.unsqueeze(0).unsqueeze(0)
    rhotensor = rhotensor.to(device="cuda")

    rho_pad = torch.nn.functional.pad(rhotensor, (pad_size, pad_size,pad_size,pad_size), mode="circular")
    rho_pad = rho_pad.squeeze(0).squeeze(0)
    unfolded_rho = rho_pad.unfold(0, window_dim,stride ).unfold(1, window_dim, stride)
    values = mulocmatrix[::stride,::stride]
    windows = unfolded_rho.contiguous().view(-1, window_dim, window_dim)
    windows = windows.to(device="cpu")
    values = torch.tensor(values, dtype=torch.float32)
    values = values.flatten()
    return windows,values

def cut_density_windows_torch_padded_modforsmallgpu(rhomatrix,mulocmatrix, window_dim):
    stride = 1
    pad_size = window_dim//2
    rhotensor = torch.tensor(rhomatrix, dtype=torch.float32)
    rhotensor = rhotensor.unsqueeze(0).unsqueeze(0)
    #rhotensor = rhotensor.to(device="cuda")
    print("rhotensor shape",rhotensor.shape)
    print("pad_size",pad_size)
    rho_pad = torch.nn.functional.pad(rhotensor, (pad_size, pad_size,pad_size,pad_size), mode="circular")
    rho_pad = rho_pad.squeeze(0).squeeze(0)
    unfolded_rho = rho_pad.unfold(0, window_dim,stride ).unfold(1, window_dim, stride)
    values = mulocmatrix[::stride,::stride]
    windows = unfolded_rho.contiguous().view(-1, window_dim, window_dim)
    windows = windows.to(device="cpu")
    values = torch.tensor(values, dtype=torch.float32)
    values = values.flatten()
    return windows,values



def build_training_data(df, width,L,window_stack,center_values):
    n_windows = int(L/width)
    window_dim = int(np.sqrt(len(df))/n_windows)
    window,center = cut_density_windows(df, window_dim,n_windows)
    return window_stack,center_values

def build_training_data_torch(df, width,L):
    n_windows = int(L/width)
    
    window_dim = int(np.sqrt(len(df))/n_windows)
    print("window_dim",window_dim)
    window_dim = int(np.sqrt(len(df))*width/L)
    print("window_dim",window_dim)

    window,center = cut_density_windows_torch_padded(df, window_dim)
    return window,center

def build_training_data_torch_optimized(df, width,L):
    n_windows = int(L/width)
    
    window_dim = int(np.sqrt(len(df))/n_windows)
    print("window_dim",window_dim)
    window_dim = int(np.sqrt(len(df))*width/L)
    print("window_dim",window_dim)
    rhomatrix = df.pivot(index='y', columns='x', values='rho').values
    mulocmatrix = df.pivot(index='y', columns='x', values='muloc').values
    shape = rhomatrix.shape
    dim = shape[0]
    for i in range(2):
        for j in range(2):
            rhomatrixsmall=rhomatrix[i*int(dim/2):(i+1)*int(dim/2),j*int(dim/2):(j+1)*int(dim/2)] 
            mulocmatrixsmall=mulocmatrix[i*int(dim/2):(i+1)*int(dim/2),j*int(dim/2):(j+1)*int(dim/2)] 
            window,center = cut_density_windows_torch_padded_modforsmallgpu(rhomatrix,mulocmatrix, window_dim)
    return window,center

def reconstruct_values(values, stride, window_dim):
    n_windows = int(np.sqrt(len(values)))
    values = values.view(n_windows, n_windows)
    print("values shape",values.shape)
    values = values.repeat_interleave(stride, dim=0).repeat_interleave(stride, dim=1)
    print("values shape",values.shape)
#    values = values[:window_dim, :window_dim]
    print("values shape",values.shape)
    return values


def neural_c1(model,rhomatrix):
    """
    This function takes a model and a rhomatrix (not too big) as input, and returns the neural c1 value.
    """
    _,_,window_dims,_ =next(model.children()).weight.shape
    print("rhoshape",rhomatrix.shape)
    window,_ = cut_density_windows_torch_padded_modforsmallgpu(rhomatrix,rhomatrix,window_dims)
    model.eval()
    outputs= []
    with torch.no_grad():
        for input in window:
            input = input.unsqueeze(0)
            input = input.to(torch.device("cuda"))
            output = model(input)
            output = output.cpu().numpy()
            outputs.append(output)

    c1_reconstruct = reconstruct_values(torch.tensor(np.array(outputs)), 1,window_dims)
    return c1_reconstruct

def picard_minimization(L,mu,T,dx,model,nperiod,Amp,alpha=0.3,max_iter=3000):
    x = np.linspace(0,L,int(L/dx))
    y = np.linspace(0,L,int(L/dx))
    xg, yg = np.meshgrid(x, y)
    rho = np.zeros((len(x), len(y)))
    rhobuffer = rho.copy()
    rho = rho + 0.01
    delta = 1
    i = 0
    while delta > 0.1:
        rhobuffer =np.exp(-potential(xg,nperiod,L,Amp)/T+mu/T)+np.array(neural_c1(model,rho)) 
        rho = (1-alpha)*rho + alpha*rhobuffer
        delta = np.max(np.abs(rho - rhobuffer))
        i += 1
        print("Iteration: ", i)
        print("Delta: ", delta)
        if i > max_iter:
            print("Max iterations reached")
            return 
    return rho,xg,yg

def c2(model, rhomatrix, dx=0.01):
    _,_,window_dims,_ =next(model.children()).weight.shape
    print("rhoshape",rhomatrix.shape)
    windows,_ = cut_density_windows_torch_padded_modforsmallgpu(rhomatrix,rhomatrix,window_dims)
    print("windows shape",windows.shape)
    windows = windows.detach().requires_grad_(True)
    jacobiWindows = vmap(jacrev(model), in_dims=(0,))(windows)
    print("jacobiWindows shape",jacobiWindows.shape)
    # c1_result = result.numpy().flatten()
    # if c2 == "unstacked":
    #     return c1_result, jacobiWindows
    # c2_result = np.row_stack([np.roll(np.pad(jacobiWindows[i], (0,rho.shape[0]-inputBins)), i-windowBins) for i in range(rho.shape[0])])
    # return c1_result, c2_result


def neural_c2(model,rhomatrix):
    """
    This function takes a model and a rhomatrix (not too big) as input, and returns the neural c1 value.
    """
    _,_,window_dims,_ =next(model.children()).weight.shape
    print("rhoshape",rhomatrix.shape)
    window,_ = cut_density_windows_torch_padded_modforsmallgpu(rhomatrix,rhomatrix,window_dims)
    print("window shape",window.shape)
    window = window.detach().requires_grad_(True)
    model.eval()
    outputs= []
    for input in window:
        input = input.unsqueeze(0)
        input = input.to(torch.device("cuda"))
        jacobian = torch.autograd.functional.jacobian(model, input)
        jacobian = jacobian.squeeze().cpu().numpy()
        outputs.append(jacobian)

    print("outputs shape",np.array(outputs).shape)

def potential(x,nperiod,L,Amp):
    Lperiod = L/nperiod
    V = Amp*np.cos(2*np.pi*x/Lperiod)
    return V