import torch
import numpy as np
import os
import gc
from data_pipeline.create_training_data_func import _extract_windows_inference




# def neural_c1(model, rho,num_slices,sim_L, dx,c2=False):
def neural_c1(model, rho,num_slices,sim_L, dx):
    ## rho matrix is either list of single matrix (as torch tensor) or a list of matrices
    _,_,window_dims,_ =next(model.children()).weight.shape
    slice_length= int(rho.shape[0] / num_slices)
    result_length= slice_length - window_dims + 1
    #N = rho_matrix.shape[0]
    #output_dim = N - window_dims + 1
    # output=torch.empty((output_dim, output_dim),device="cpu")
    # rho_matrix=rho_matrix.to("cuda")
    outputs=[]
    model.eval()
    
    if num_slices == 1:
            slice_output=[]
            rho = rho.cuda(non_blocking=True)
            windows,num_windows = _extract_windows_inference(1, window_dims, rho)
            print("predict")
            print(f"Number of windows: {num_windows}")
            for k in range(num_windows):
                # print("window shape", windows[k,:,:,:].shape)
                inputs = windows[k,:,:,:]
                output = model(inputs).view(-1).cpu()
                slice_output.append(output)
            slice_output = torch.cat(slice_output, dim=0).reshape(result_length, result_length)
            # rho_slice = rho_slice.cpu()
            # diff=(slice_output-rho_slice)
            # assert torch.sum(diff) == 0, f"Output does not match input slice at position ({i},{j}). Difference: {torch.sum(diff)}"
            del windows
            torch.cuda.empty_cache()
            gc.collect()
            return slice_output.detach().numpy()


    for i in range(num_slices):
        for j in range(num_slices):
            print(f"Processing slice {i+1}/{num_slices}, {j+1}/{num_slices}.")
            slice_output=[]
            startx = i * slice_length
            endx = (i + 1) * slice_length
            starty = j * slice_length
            endy = (j + 1) * slice_length
            rho_slice = rho[startx:endx, starty:endy]
            print("rho_slice shape", rho_slice.shape)
            rho_slice = rho_slice.cuda(non_blocking=True)
            windows,num_windows = _extract_windows_inference(1, window_dims, rho_slice)
            print("predict")
            print(f"Number of windows: {num_windows}")
            for k in range(num_windows):
                # print("window shape", windows[k,:,:,:].shape)
                inputs = windows[k,:,:,:]
                output = model(inputs).view(-1).cpu()
                slice_output.append(output)
            slice_output = torch.cat(slice_output, dim=0).reshape(slice_length-window_dims+1, slice_length-window_dims+1)
            # rho_slice = rho_slice.cpu()
            # diff=(slice_output-rho_slice)
            # assert torch.sum(diff) == 0, f"Output does not match input slice at position ({i},{j}). Difference: {torch.sum(diff)}"
            outputs.append(slice_output)
            del slice_output, windows, rho_slice
            torch.cuda.empty_cache()
            gc.collect()

    outputs = torch.stack(outputs, dim=0)
    final_outputs = torch.zeros(rho.shape[0], rho.shape[1])
    for i in range(num_slices):
        for j in range(num_slices):
            startx = i * slice_length
            endx = (i + 1) * slice_length
            starty = j * slice_length
            endy = (j + 1) * slice_length
            final_outputs[startx:endx, starty:endy] = outputs[i * num_slices + j] 
    # diff = final_outputs - rho
    # assert torch.sum(diff) == 0, f"Output does not match input at the end. Difference: {torch.sum(diff)}"
    return final_outputs.numpy()


def neural_c2(model, rho,num_slices,sim_L, dx,y_norm=False):

    ## output (if norm is true) is 3D tensor, where first dimension is 
    ## rho matrix is either list of single matrix (as torch tensor) or a list of matrices
    # _,_,window_dims,_ =next(model.children()).weight.shape
    window_dims=3
    slice_length= int(sim_L*dx / num_slices)
    rho_x = rho.shape[0]
    rho_y = rho.shape[1]
    pad_size = rho_x - window_dims
    even_uneven = 1 - rho_x % 2 
    #N = rho_matrix.shape[0]
    #output_dim = N - window_dims + 1
    # output=torch.empty((output_dim, output_dim),device="cpu")
    # rho_matrix=rho_matrix.to("cuda")
    outputs=[]
    model.eval()
    device= torch.device("cuda")
    model = model.to(device)
    if y_norm:
        slice_output=[]
        rho_slice=rho[rho_y//2-window_dims//2:rho_y//2+1+window_dims//2,:]  # Use only middle slice of rho in y direction for c2 computation due to translational symmetry
        rho_slice = rho_slice.cuda(non_blocking=True)
        windows,num_windows = _extract_windows_inference(1, window_dims, rho_slice,c2=True)
        print("predict")
        print(f"Number of windows: {num_windows}")
        assert num_windows == rho_x, "Number of windows does not match the expected number based on rho_x."
        windows = windows.detach().requires_grad_(True)
        for k in range(num_windows):
            jacobiWindows =torch.autograd.functional.jacobian(model,windows[k,:,:,:]).squeeze(0).squeeze(0).squeeze(0).squeeze(0).cpu()
            print("jacobiWindows shape", jacobiWindows.shape)
            padded_jac = torch.nn.functional.pad(jacobiWindows, (0,pad_size,pad_size//2+even_uneven,pad_size//2), mode='constant', value=0)
            assert padded_jac.shape == rho.shape, f"Expected padded_jac shape to be {rho.shape}, but got {padded_jac.shape}."
            print("padded_jac shape", padded_jac.shape)
            padded_jac_switch = torch.roll(padded_jac, shifts=(k-window_dims//2,0), dims=(1, 0))
            print(padded_jac_switch)
            slice_output.append(padded_jac_switch)

        slice_output = torch.stack(slice_output, dim=0).reshape(rho_x,rho.shape[0], rho.shape[1])
        check = slice_output[-1,:,:]- padded_jac_switch
        assert slice_output.shape == (rho_x, rho.shape[0], rho.shape[1]), f"Expected slice_output shape to be {(rho_x, rho.shape[0], rho.shape[1])}, but got {slice_output.shape}."
        assert torch.sum(check) == 0, f"Output does not match input slice at the end. Difference: {torch.sum(check)}"
        return slice_output
    else:
        for i in range(num_slices):
            for j in range(num_slices):
                print(f"Processing slice {i+1}/{num_slices}, {j+1}/{num_slices}.")
                slice_output=[]
                startx = i * slice_length
                endx = (i + 1) * slice_length
                starty = j * slice_length
                endy = (j + 1) * slice_length
                rho_slice = rho[startx:endx, starty:endy]
                rho_slice = rho_slice.cuda(non_blocking=True)
                windows,num_windows = _extract_windows_inference(1, window_dims, rho_slice)
                print("predict")
                windows = windows.detach().requires_grad_(True)
                for k in range(num_windows):
                    x_ind = k // (slice_length  ) + i**slice_length
                    y_ind = k % (slice_length ) + j**slice_length
                    jacobiWindows =torch.autograd.functional.jacobian(model,windows[k,:,:,:]).squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                    padded_jac = torch.nn.functional.pad(jacobiWindows, (0,pad_size,0,pad_size), mode='constant', value=0)
                    padded_jac_switch = torch.roll(padded_jac, shifts=(x_ind-window_dims//2,y_ind-window_dims//2), dims=(0, 1))
                    slice_output.append(padded_jac_switch)
                
                slice_output = torch.stack(slice_output, dim=0).reshape(slice_length, slice_length,rho.shape[0], rho.shape[1])
                outputs.append(slice_output)

        
        final_outputs = torch.zeros(rho.shape[0], rho.shape[1], rho.shape[0], rho.shape[1])
        for i in range(num_slices):
            for j in range(num_slices):
                startx = i * slice_length
                endx = (i + 1) * slice_length
                starty = j * slice_length
                endy = (j + 1) * slice_length
                final_outputs[startx:endx, starty:endy,:,:] = outputs[i * num_slices + j]
        print("outputs shape", final_outputs.shape)
        for i in range(final_outputs.shape[0]):
            for j in range(final_outputs.shape[1]):
                print(f"Output at ({i},{j}): {final_outputs[i,j,:,:]}")
        print(outputs[-1][-1,-1,:,:])
        return final_outputs.numpy()

def static_structure_factor(c2):
    ## use y symmetry of c2 to reduce dimensions
    red_y =  c2[:,]


def lattice_fourier_transform(c2,period_length, dx,num_modes):
    """
    Compute the lattice Fourier transform of the c2 tensor.
    """
    a = period_length
    # Assuming c2 is a 3D tensor with shape (x,x',|z'|)
    ## first do regular fft for z dimension (direction without potential)
    fourier_c2 = np.fft.rfft(c2, axis=2)
    c2_mu_nu = np.zeros((num_modes, num_modes, fourier_c2.shape[2]))
    ### then do lattice Fourier transform for x and x' dimensions
    for mu in range(num_modes):
        Q_mu = 2* np.pi * mu / a
        for nu in range(num_modes):
            Q_nu = 2* np.pi * nu / a
        
def picard_minimization(L,mu,T,model,nperiod,Amp,alpha=0.3,max_iter=1000):
    _,_,window_dims,_ =next(model.children()).weight.shape

    # x = np.linspace(0,L,int(L/dx))
    # y = np.linspace(0,L,)
    # xg, yg = np.meshgrid(x, y)
    resolution = 50
    x = np.linspace(0,10,resolution*L)
    y = np.linspace(0,5,window_dims)
    xg, yg = np.meshgrid(x, y)
    assert xg.shape[1] == resolution*L, "xg shape mismatch"
    assert xg.shape[0] == window_dims, "yg shape mismatch"
    rho = np.zeros((window_dims, resolution*L))
    rhobuffer = rho.copy()
    rho = rho + 0.01
    delta = 1
    i = 0
    Lperiod = L/nperiod
    V = Amp*np.cos(2*np.pi*xg/Lperiod)
    muloc=(V-mu)/T
    while delta > 0.1:
        rhobuffer =np.exp(symmetry_neural_c1(model,rho)-muloc) 
        rho = (1-alpha)*rho + alpha*rhobuffer
        delta = np.max(np.abs(rho - rhobuffer))
        i += 1
        print("Iteration: ", i)
        print("Delta: ", delta)
        if i > max_iter:
            print("Max iterations reached")
            return 
    c1 = symmetry_neural_c1(model,rho)
    return rho, c1

def symmetry_neural_c1(model, rho):
    ## rho matrix is numpy array
    rho = torch.tensor(rho, dtype=torch.float32)    
    _,_,window_dims,_ =next(model.children()).weight.shape
    rho_x = rho.shape[1]
    rho_y = rho.shape[0]
    assert window_dims == rho_y, "rho must be linear with window dimensions"
    result_length= rho_x - window_dims + 1
    #N = rho_matrix.shape[0]
    #output_dim = N - window_dims + 1
    # output=torch.empty((output_dim, output_dim),device="cpu")
    # rho_matrix=rho_matrix.to("cuda")
    outputs=[]
    model.eval()
    
    slice_output=[]
    rho = rho.cuda(non_blocking=True)
    # print(rho.shape)
    # rho = rho.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    # rho = torch.nn.functional.pad(rho, (window_dims//2,window_dims//2,window_dims//2,window_dims//2), mode='circular')
    # rho = rho.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    windows,num_windows = _extract_windows_inference(1, window_dims, rho,padding=True)
    print("predict")
    print(f"Number of windows: {num_windows}")
    for k in range(num_windows):
        # print("window shape", windows[k,:,:,:].shape)
        inputs = windows[k,:,:,:]
        output = model(inputs).view(-1).cpu()
        slice_output.append(output)
    slice_output = torch.cat(slice_output, dim=0).reshape(rho_y, rho_x)
    # rho_slice = rho_slice.cpu()
    # diff=(slice_output-rho_slice)
    # assert torch.sum(diff) == 0, f"Output does not match input slice at position ({i},{j}). Difference: {torch.sum(diff)}"
    del windows
    torch.cuda.empty_cache()
    gc.collect()
    return slice_output.detach().numpy()


    