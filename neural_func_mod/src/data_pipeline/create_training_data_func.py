import torch
import numpy as np 
import pandas as pd
import os
import sys
# import gc
import utils.neural_helper as helper
from utils.set_paths import set_paths
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

import torch.nn.functional as F
from memory_profiler import profile

#@profile
def load_training_data(names,window_L,sim_L=15,n_profiles=-1,mod=True):

    rho_list = []
    c1_list = []
    for i in names:
        df,extra = helper.load_df(i)
        # Get the number of rows and columns in the DataFrame
        if not mod:
            L_period_perturb = 15/extra["period_perturb"]
            df["muloc"] = df["muloc"]+ extra["amp_perturb"]* np.sin(2 * np.pi * df["y"] / L_period_perturb)
        df.loc[df["rho"] <= 0, "rho"] = 1e-10
        df["muloc"]=np.log(df["rho"])+df["muloc"]
        if np.sum(df["rho"])< 0.1:
            continue
        
        rhomatrix = df.pivot(index='y', columns='x', values='rho').values
        mulocmatrix = df.pivot(index='y', columns='x', values='muloc').values
        rhotensor = torch.tensor(rhomatrix, dtype=torch.float32)#.to("cuda")
        c1tensor = torch.tensor(mulocmatrix, dtype=torch.float32)#.to("cuda")
        rho_list.append(rhotensor)
        c1_list.append(c1tensor)
    shape = rhomatrix.shape
    dim = shape[0]
    dx = dim / sim_L
    window_dim = int(dx*window_L)
    if window_dim % 2 == 0:
        window_dim += 1
    return rho_list, c1_list, window_dim, dx


def plot_density_profiles_2d_3d_debug(tag,num=-1,start=0):
    # Path to the Density_profiles directory
    plotname = tag + "dens_plot.png"
    data_dir, density_profiles_dir, neural_func_dir = set_paths()

    savename = os.path.join(data_dir,"data_generation/density_plots",plotname)
# Path to the Density_profiles directory
    density_profiles_dir = os.path.join(density_profiles_dir, tag)

    names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]
    rho_list = []
    c1_list = []

    slice = names[start:num]
    rows =len(slice)
    cols = 2
    ## create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(11,10*rows/2))
    fig.suptitle("2D Density profile")
    
    for i, name in enumerate(slice):
        # Get the current dataframe
        df,extra = helper.load_df(name,density_profiles_dir)
        # Get the number of rows and columns in the DataFrame
        L_period_perturb = 15/extra["period_perturb"]
        df["muloc"] = df["muloc"]+ extra["amp_perturb"]* np.sin(2 * np.pi * df["y"] / L_period_perturb)
        df["muloc"]=np.log(df["rho"])+df["muloc"]
        if np.sum(df["rho"])< 0.1:
            continue
        label = f"Nperiod = {extra["nperiod"]}, mu = {extra["mu"]}, packing = {extra["packing"]}, amp = {extra["amp"]}"
        shape = np.sqrt(len(df)).astype(int)
        xi = df["x"].values.reshape(shape, shape)
        yi = df["y"].values.reshape(shape,shape)
        rho = df["rho"].values.reshape(shape,shape)
        c1 = df["muloc"].values.reshape(shape,shape)
        pcm = axs[i,0].pcolormesh(xi, yi, rho, shading='auto', cmap='viridis',label=label)
        axs[i,0].set_xlabel(r"$x/\sigma$")
        axs[i,0].set_ylabel(r"$y/\sigma$")
        patch = mpatches.Patch(color=pcm.cmap(0.5), label=label)
        axs[i,0].legend(handles=[patch])

        x_values = df.groupby("x").mean().reset_index()["x"]
        rho_values = df.groupby("x").mean().reset_index()["rho"]
        pcm2=axs[i,1].pcolormesh(xi, yi, c1, shading='auto', cmap='viridis',label=label)
        axs[i,1].set_xlabel(r"$x/\sigma$")
        axs[i,1].set_ylabel(r"$y/\sigma$")
        axs[i,1].set_title(slice[i])



        fig.colorbar(pcm, ax=axs[i,0], label=r"$\rho^{(1)}(\mathbf{x})$") 
        fig.colorbar(pcm2, ax=axs[i,1], label=r"$c_1(\mathbf{x})$") 
    plt.tight_layout()
    plt.savefig(savename, dpi=300)

@profile
def load_data_chunk(path):
    """
    Loads prepared chunks of windows and labels from all .pt files in the specified path.
    Args:
        path (str): Path to the data file.
    Returns:
        list of dictionaries: Each dictionary contains 'windows' and 'labels' tensors.
    """
    names= [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    data_chunks = []
    for name in names:
        file_path = os.path.join(path, name)
        data = torch_load(file_path)
        data_chunks.append(data)
    window_dim = data_chunks[0]['windows'].shape[2]  # Assuming all chunks have the same window size
    return data_chunks, window_dim

@profile
def torch_load(path):
    """
    Loads a .pt file from the specified path.
    Args:
        path (str): Path to the .pt file.
    Returns:
        dict: Loaded data containing 'windows' and 'labels'.
    """
    print(f"Loading data from {path}")
    data = torch.load(path)
    print(data['windows'].shape, data['labels'].shape)
    return data

def _extract_windows_inference(stride,window_size, x,c2=False):
    """
    x, y: single matrix pair on GPU, shape (H, W)
    Returns:
        windows: [L, 1, win, win]
        labels: [L]
    """
    # x = x.cuda(non_blocking=True)  # Ensure x is on GPU
    # y = y.cuda(non_blocking=True)  # Ensure y is on GPU
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    win = window_size
    stri = stride

    # Unfold to get all windows as columns
    pad_size = win //2
    # Apply padding to ensure windows can be extracted correctly
    if c2:
        x = F.pad(x, (pad_size,pad_size,0,0), mode='circular') # Reflect padding
    #else:
        #sbax = F.pad(x, (pad_size,pad_size,pad_size,pad_size), mode='circular')  # Reflect padding
    #print(x.shape)
    unfolded = F.unfold(x, kernel_size=win, stride=stri)  # [1, win*win, L]

    num_windows = unfolded.shape[-1]
    # Reshape to [L, 1, win, win]
    windows = unfolded.transpose(1, 2).reshape(num_windows, 1, win, win)
    # windows_y = unfoldedy.transpose(1, 2).reshape(num_windows, 1, win, win)

    
    return windows,num_windows

# @profile
def _extract_windows_unfold(stride,window_size, x, y,pooling=False,padding=True):
    """
    x, y: single matrix pair on GPU, shape (H, W)
    Returns:
        windows: [L, 1, win, win]
        labels: [L]
    """
    # x = x.cuda(non_blocking=True)  # Ensure x is on GPU
    # y = y.cuda(non_blocking=True)  # Ensure y is on GPU
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    win = window_size
    stri = stride
    if pooling:
        y = y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # [1, 1, H', W']
        y = F.avg_pool2d(y, kernel_size=2, stride=2)  # [1, 1, H', W']
        win = win // 2  # Adjust window size for pooling
        # y_t=y
        y = y.squeeze(0).squeeze(0)  # Remove batch and channel dimensions for labels
    # Unfold to get all windows as columns
    pad_size = win //2
    if padding:
        # Apply padding to ensure windows can be extracted correctly
        x = F.pad(x, (pad_size,pad_size,pad_size,pad_size,), mode='circular')  # Reflect padding
        # y_t = F.pad(y_t, (pad_size,pad_size,pad_size,pad_size,), mode='circular')  # Reflect padding
        unfolded = F.unfold(x, kernel_size=win, stride=stri)  # [1, win*win, L]
        # unfoldedy = F.unfold(y_t, kernel_size=win, stride=stri)  # [1, win*win, L]

        num_windows = unfolded.shape[-1]
        # Reshape to [L, 1, win, win]
        windows = unfolded.transpose(1, 2).reshape(num_windows, 1, win, win)
        # windows_y = unfoldedy.transpose(1, 2).reshape(num_windows, 1, win, win)


        H = x.shape[-2]
        W = x.shape[-1]
        rows = torch.arange(0, H - win + 1, stride, device=x.device)
        cols = torch.arange(0, W - win + 1, stride, device=x.device)
        ii, jj = torch.meshgrid(rows, cols, indexing='ij')

        center_i = ii 
        center_j = jj 
        labels = y[center_i, center_j].reshape(-1)
        ## check if labels are correct
        # for i in range(num_windows):
            # assert labels2[i]==windows_y[i,0,2,2] , "proper missmatch"
            # if labels[i] != labels2[i]: 
            #     print(f"Label mismatch at index {i}: {labels[i]} != {labels2[i]}")
            #     print(windows_y[i])
            #     print(labels[i],labels2[i],windows_y[i,0,2,2])
                # exit()
        return windows, labels
    else:
        unfolded = F.unfold(x, kernel_size=win, stride=stri)  # [1, win*win, L]
        num_windows = unfolded.shape[-1]
        windows = unfolded.transpose(1, 2).reshape(num_windows, 1, win, win)
        # Compute label positions (center of each window)
        H = x.shape[-2]
        W = x.shape[-1]
        rows = torch.arange(0, H - win + 1, stride, device=x.device)
        cols = torch.arange(0, W - win + 1, stride, device=x.device)
        ii, jj = torch.meshgrid(rows, cols, indexing='ij')

        center_i = ii + pad_size
        center_j = jj + pad_size
        labels = y[center_i, center_j].reshape(-1)  # [L]
        return windows, labels


# @profile
def extract_windows_from_chunk( stride,window_size,density_chunk, label_chunk,pooling=False,cat_CPU=False,padding=True):
    all_windows, all_labels = [], []
    count = 0
    for x_cpu, y_cpu in zip(density_chunk, label_chunk):
        # Move to GPU first
        x = x_cpu.cuda(non_blocking=True)
        y = y_cpu.cuda(non_blocking=True)

        windows, labels =_extract_windows_unfold(stride,window_size,x, y,pooling=pooling,padding=padding)
        print("extracted windows")
        if cat_CPU:
            # Move back to CPU if needed
            windows = windows.cpu()
            labels = labels.cpu()  # Move back to CPU

        all_windows.append(windows)
        all_labels.append(labels)
        count += 1

    windows = torch.cat(all_windows, dim=0)
    labels = torch.cat(all_labels, dim=0)

    data = {
        'windows': windows.cpu(),
        'labels': labels.cpu()
    } 
    return data