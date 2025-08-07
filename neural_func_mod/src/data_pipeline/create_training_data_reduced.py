import numpy as np 
import torch
import os
import random
import sys
import gc
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
mod = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
stri = int(sys.argv[7]) if len(sys.argv) > 7 else 6


def gen_data(sim_L,window_L,names, chunk_size,CPU_cat,stride,output_dir,num_slices):
    size_file = os.path.join(output_dir, "size.txt")
    num_chunks = int(np.ceil(len(names)/chunk_size))
    name_chunk_list = []
    for i in range(num_chunks):
        start_idx = i*chunk_size
        end_idx = min((i+1)*chunk_size, len(names))
        name_chunk_list.append(names[start_idx:end_idx])
    ## loop through chunks, divide each into smaller matrices, take windows from those and save them then
    for i in range(num_chunks):
        print(f"Processing chunk {i+1}/{num_chunks} with {len(name_chunk_list[i])} profiles.")
        rho_chunk_list, c1_chunk_list, window_dim, dx = load_training_data(name_chunk_list[i],window_L=window_L,sim_L=sim_L,mod=mod)
        slice_length= int(sim_L*dx / num_slices)
        print(window_dim)
        print(len(rho_chunk_list),len(c1_chunk_list),chunk_size)
        #assert len(rho_chunk_list) == chunk_size, "Mismatch in chunk lengths"
        slice_count=0
        for j in range(num_slices):
            for k in range(num_slices):
                # print(f"Processing slice {(slice_count)+1}/{num_slices**2} in chunk {i+1}/{num_chunks}.")
                startx = j * slice_length
                endx = (j + 1) * slice_length
                starty = k * slice_length
                endy = (k + 1) * slice_length
                #### get lists with three tensors, each the same slize of rho and c1 in chunk
                s = [rho_chunk_list[l][startx:endx,starty:endy] for l in range(len(rho_chunk_list))]
                s_c1 = [c1_chunk_list[l][startx:endx,starty:endy] for l in range(len(c1_chunk_list))]
                print(s[0].shape, s_c1[0].shape)

                #print("length of slice",len(s),len(s_c1))
                ### check if the slice is valid
                if len(s) == 0:
                    print(f"Skipping chunk {i}, slice {j} due to empty slice.")
                    continue
                # if len(s[0]) < window_dim:
                #     print(f"Skipping chunk {i}, slice {j} due to insufficient length.")
                #     continue
                 
                if len(s) < chunk_size:
                    print(f"Not enough profiles in chunk {i}, slice {j}. Skipping.")
                    continue
                if len(s) % chunk_size != 0:
                    print(f"Chunk size mismatch in chunk {i}, slice {j}. Skipping.")
                    continue
                 
                ## Extract windows from the slices of chunk, and then directly save   

                data = extract_windows_from_chunk(stride,window_dim,s,s_c1,pooling=False,cat_CPU=CPU_cat,padding=True)
                
                savename=f"window_chunk_{i}_slice_{slice_count}.pt"
                print(savename)
                print(os.path.join(output_dir, savename))
                if not os.path.exists(output_dir):
                    print("damn")
                    exit()
                len_data = len(data['windows'])
                with open(size_file, 'a') as f:
                    f.write(f"{savename} ;{len_data}\n")
                torch.save(data, os.path.join(output_dir, savename))
                del data,s,s_c1
                gc.collect()
                torch.cuda.empty_cache()
                slice_count += 1
        del rho_chunk_list, c1_chunk_list
        gc.collect()

    print(f"{output_dir}")


sim_L = 15
cpu_cat = [True]
chunk_size=range(5,13)
# chunk_size=[11,12]

data_dir, density_profiles_dir, neural_func_dir = set_paths()

## Prepare data paths
timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(data_dir,"datasets",timestamp+tag)
os.makedirs(output_dir, exist_ok=True)
names= helper.get_names(density_profiles_dir,tag,ending=".dat")
# RAMfile = os.path.join(output_dir, "RAM_usage.txt")
# RAbatfile2 = os.path.join(output_dir, "RAM_usage.txt")
old_sub_output_dir = None
for cpu in [False,True]:
    for s in [6,5,4]:
        for c_s in chunk_size:
            sub_output_dir = os.path.join(output_dir,f"stride_{s}_catCPU_{cpu}_chunk_{c_s}")
            os.makedirs(sub_output_dir, exist_ok=True)
            print(f"Output directory created: {sub_output_dir}")
            try:
                gen_data(sim_L,window_L,names, c_s,cpu,s,sub_output_dir,num_slices)
                try:
                    if os.path.exists(old_sub_output_dir):
                        print(f"Removing old output directory: {old_sub_output_dir}")
                        os.system(f"rm -rf {old_sub_output_dir}")
                except Exception as e:
                    print("No old output directory to remove or error:")
                old_sub_output_dir = sub_output_dir

            except Exception as e:
                print(f"Error processing stride {s}, cat_CPU {cpu}, chunk_size {c_s}: {e}")
                print("Removing incomplete output directory:", sub_output_dir)
                if os.path.exists(sub_output_dir):
                    os.system(f"rm -rf {sub_output_dir}")
                break

