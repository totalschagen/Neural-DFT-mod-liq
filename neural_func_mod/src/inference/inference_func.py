import pytorch as torch
import numpy as np
import os

def neural_c1(model, rho_matrix,batch_size=64):
    ## rho matrix is either list of single matrix (as torch tensor) or a list of matrices
    _,_,window_dims,_ =next(model.children()).weight.shape
    N = rho_matrix.shape[0]
    output_dim = N - window_dims + 1
    output=torch.empty((output_dim, output_dim),device="cpu")
    rho_matrix=rho_matrix.to("cuda")
    for i in range(output_dim):
        windows=[]
        for j in range(output_dim):
            window = rho_matrix[i:i+window_dims,j:j+window_dims].unsqueeze(0).unsqueeze(0)
            windows.append(window)
            if len(windows) == batch_size or  j == output_dim - 1:
                windows_tensor = torch.cat(windows, dim=0)
                pred = model(windows_tensor).view(-1).detach().cpu()
                output[i, j-len(windows)+1:j+1] = pred
                windows = []
    return output.numpy()

