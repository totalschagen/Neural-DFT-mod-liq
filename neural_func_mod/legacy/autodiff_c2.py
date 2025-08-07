import torch
import numpy as np
import os 
import neural_helper as helper
from torch.func import vmap,jacrev
def c2(model, rhomatrix, dx=0.01):
    _,_,window_dims,_ =next(model.children()).weight.shape
    print("rhoshape",rhomatrix.shape)
    windows,_ = helper.cut_density_windows_torch_padded_modforsmallgpu(rhomatrix,rhomatrix,window_dims)
    windows = windows.detach().requires_grad_(True)
    jacobiWindows = vmap(jacrev(model), in_dims=(0,))(windows)
    print("jacobiWindows shape",jacobiWindows.shape)
    # c1_result = result.numpy().flatten()
    # if c2 == "unstacked":
    #     return c1_result, jacobiWindows
    # c2_result = np.row_stack([np.roll(np.pad(jacobiWindows[i], (0,rho.shape[0]-inputBins)), i-windowBins) for i in range(rho.shape[0])])
    # return c1_result, c2_result

def neural_c1(model,rhomatrix):
    """
    This function takes a model and a rhomatrix (not too big) as input, and returns the neural c1 value.
    """
    _,_,window_dims,_ =next(model.children()).weight.shape
    print("rhoshape",rhomatrix.shape)
    window,_ = helper.cut_density_windows_torch_padded_modforsmallgpu(rhomatrix,rhomatrix,window_dims)
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

