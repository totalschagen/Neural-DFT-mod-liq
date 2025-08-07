import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
import random
import sys
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import models.conv_network as net
import utils.neural_helper as helper
from utils.set_paths import set_paths
from data_pipeline.create_training_data_func import load_training_data,extract_windows_from_chunk,_extract_windows_unfold
from data_pipeline.data_loader_class import ChunkManager, SlidingWindowDataset

import torch.nn.functional as F

tag = str(sys.argv[1])
window_L = float(sys.argv[2]) 
num_profiles = int(sys.argv[3]) if len(sys.argv) > 3 else -1
chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else 3
num_slices = int(sys.argv[5]) if len(sys.argv) > 5 else 1

data_dir, density_profiles_dir, neural_func_dir = set_paths()

## Prepare data paths
timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(data_dir,"datasets",timestamp+tag)
os.makedirs(output_dir, exist_ok=True)
sim_L = 15
### load and prepare data set
"""
rho_list,c1_list,window_dim,dx= load_training_data(tag,window_L=window_L,sim_L=sim_L,n_profiles=num_profiles)
### divide data into chunks
num_chunks = int(np.ceil(len(rho_list)/chunk_size))
rho_chunk_list = []
c1_chunk_list = []
for i in range(num_chunks):
    start_idx = i*chunk_size
    end_idx = min((i+1)*chunk_size, len(rho_list))
    rho_chunk_list.append(rho_list[start_idx:end_idx])
    c1_chunk_list.append(c1_list[start_idx:end_idx])
## loop through chunks, divide each into smaller matrices, take windows from those and save them then

slice_length= int(sim_L*dx / num_slices)
for i in range(num_chunks):
    print(f"Processing chunk {i+1}/{num_chunks} with {len(rho_chunk_list[i])} profiles.")
    slice_count=0
"""
names= helper.get_names(density_profiles_dir,tag)
#rho_list,c1_list,window_dim,dx= load_training_data(tag,window_L=window_L,sim_L=sim_L,n_profiles=num_profiles)
### divide data into chunks
num_chunks = int(np.ceil(len(names)/chunk_size))
name_chunk_list = []
for i in range(num_chunks):
    start_idx = i*chunk_size
    end_idx = min((i+1)*chunk_size, len(names))
    name_chunk_list.append(names[start_idx:end_idx])
## loop through chunks, divide each into smaller matrices, take windows from those and save them then

for i in range(num_chunks):
    print(f"Processing chunk {i+1}/{num_chunks} with {len(name_chunk_list[i])} profiles.")
    rho_chunk_list, c1_chunk_list, window_dim, dx = load_training_data(name_chunk_list[i],window_L=window_L,sim_L=sim_L)
    slice_length= int(sim_L*dx / num_slices)
    print(len(rho_chunk_list),len(c1_chunk_list),chunk_size)
    assert len(rho_chunk_list) == chunk_size, "Mismatch in chunk lengths"
    slice_count=0
    for j in range(num_slices):
        for k in range(num_slices):
            #print(f"Processing slice {(slice_count)+1}/{num_slices**2} in chunk {i+1}/{num_chunks}.")
            start = j * slice_length
            end = (j + 1) * slice_length

            #### get lists with three tensors, each the same slize of rho and c1 in chunk
            s = [rho_chunk_list[i][l][start:end] for l in range(len(rho_chunk_list[i]))]
            s_c1 = [c1_chunk_list[i][l][start:end] for l in range(len(c1_chunk_list[i]))]
            #print("length of slice",len(s),len(s_c1))
            ### check if the slice is valid
            if len(s) == 0:
                print(f"Skipping chunk {i}, slice {j} due to empty slice.")
                continue
            if len(s[0]) < window_dim:
                print(f"Skipping chunk {i}, slice {j} due to insufficient length.")
                continue
            
            if len(s) < chunk_size:
                print(f"Not enough profiles in chunk {i}, slice {j}. Skipping.")
                continue
            if len(s) % chunk_size != 0:
                print(f"Chunk size mismatch in chunk {i}, slice {j}. Skipping.")
                continue
            
            ## Extract windows from the slices of chunk, and then directly save   
            data = extract_windows_from_chunk(2,window_dim,s,s_c1)
            savename=f"window_chunk_{i}_slice_{slice_count}.pt"
            torch.save(data, os.path.join(output_dir, savename))
            torch.cuda.empty_cache()
            slice_count += 1

print(f"saved all to Output directory: {output_dir}")