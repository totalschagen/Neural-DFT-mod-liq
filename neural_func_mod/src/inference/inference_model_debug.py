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
from inference.inference_func import picard_minimization


data_dir, density_profiles_dir, neural_func_dir = set_paths()

model_tag = str(sys.argv[1]) if len(sys.argv) > 1 else "20250730_033617chunk_training"
model_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 401


timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(data_dir,"neural_func/picard",timestamp)
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(neural_func_dir,  model_tag , "model.pth")

model = net.conv_neural_func9(model_dim)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
device= torch.device("cuda")
model = model.to(device) 
_,_, window_dim,_ = next(model.children()).weight.shape
print(f"Model loaded from {model_path} with window dimension {window_dim}")
L = 10
T = 1
nperiod=[2,5,10,15,20,30,40]
mu=[-2,-1.5,-1.0,0.0,0.5,1.0,1.5,2.0,3,4]
amp = [0.05,0.1,0.15,0.2]
for n in nperiod:
    for m in mu:
        for a in amp:
            print(f"Calculating for nperiod={n}, mu={m}, amp={a}")
            rho = picard_minimization(L=L,mu=m,T=T,model=model,nperiod=n,Amp=a)
            fig, ax = plt.subplots(1,1, figsize=(18, 6))
            a = ax.imshow(rho)
            fig.colorbar(a, ax=ax)
            save_path = os.path.join(output_dir,f"rho_n{n}_mu{m}_amp{a}.png")
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
            plt.close()


