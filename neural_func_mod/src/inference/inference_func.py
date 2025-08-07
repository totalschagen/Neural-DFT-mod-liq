import torch
import numpy as np
import os
import gc
from data_pipeline.create_training_data_func import _extract_windows_inference




# def neural_c1(model, rho,num_slices,sim_L, dx,c2=False):
def neural_c1(model, rho,num_slices,sim_L, dx):
    ## rho matrix is either list of single matrix (as torch tensor) or a list of matrices
    _,_,window_dims,_ =next(model.children()).weight.shape
    slice_length= int(sim_L*dx / num_slices)
    #N = rho_matrix.shape[0]
    #output_dim = N - window_dims + 1
    # output=torch.empty((output_dim, output_dim),device="cpu")
    # rho_matrix=rho_matrix.to("cuda")
    outputs=[]
    model.eval()

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
            slice_output = torch.cat(slice_output, dim=0).reshape(slice_length, slice_length)
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