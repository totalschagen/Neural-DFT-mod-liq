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

data_tag = "parallel2025-06-20_00-01-10"

num_slices=4
timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(data_dir,"neural_func/inference",timestamp)
os.makedirs(output_dir, exist_ok=True)
names= helper.get_names(density_profiles_dir,data_tag,ending=".dat")
model_path = os.path.join(neural_func_dir,  model_tag , "model.pth")

model = net.conv_neural_func9(401)
model.load_state_dict(torch.load(model_path))


for i in range(3):
    rho,_, window_dim, dx = load_training_data([names[i]],window_L=4.0,sim_L=15)
    rho = rho[0]  # Get the  profile
    shape = rho.shape[0]
    rho = rho[:1000,:1000]  # Take only the first half of the profile
    c2 =  neural_c2(model, rho, num_slices,10,dx)
    fourier_c2 = torch.rfftn(c2)