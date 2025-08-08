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
from data_pipeline.create_training_data_func import load_training_data
from data_pipeline.data_loader_class import prepared_windows_dataset,prepared_windows_shuffler
from inference.inference_func import neural_c1


data_dir, density_profiles_dir, neural_func_dir = set_paths()

model_tag = str(sys.argv[1]) if len(sys.argv) > 1 else "20250730_033617chunk_training"
model_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 401

data_tag = "parallel2025-08-06_16-42-14"

num_slices=1
timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(data_dir,"neural_func/inference",timestamp)
os.makedirs(output_dir, exist_ok=True)
names= helper.get_names(density_profiles_dir,data_tag,ending=".dat")
model_path = os.path.join(neural_func_dir,  model_tag , "model.pth")

model = net.conv_neural_func9(model_dim)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
device= torch.device("cuda")
model = model.to(device) 

for i in range(3):
    rho,_, window_dim, dx = load_training_data([names[i]],window_L=4.0,sim_L=15)
    rho = rho[0]  # Get the  profile
    shape = rho.shape[0]
    # rho=rho.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    # rho = torch.nn.functional.avg_pool2d(rho, kernel_size=2, stride=2)  # [1, 1, H', W']
    # rho= rho.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    rho = rho[:600,:600]  # Take only the first half of the profile
    c1 =  neural_c1(model, rho, num_slices,15,50)
    fig, ax = plt.subplots(1,2, figsize=(18, 9))
    a = ax[0].imshow(rho)
    fig.colorbar(a, ax=ax[0])
    b = ax[1].imshow(c1)
    ax[1].set_title("Network c1")
    fig.colorbar(b, ax=ax[1])
    save_path = os.path.join(output_dir,names[i]+".png")

    plt.savefig(save_path)
    plt.close()

