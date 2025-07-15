import numpy as np
import torch
import os
import models.conv_network as net
import matplotlib.pyplot as plt
import neural_helper as helper
import pandas
from utils.set_paths import set_paths
from data_pipeline.create_training_data_func import load_training_data
from inference.inference_func import neural_c1
from models.conv_network import conv_neural_func7
data_dir, density_profiles_dir, neural_func_dir = set_paths()


tag = "parallel2025-06-20_00-01-10"

rho_list,c1_list,window_dim,dx= load_training_data(tag,window_L=2.5,sim_L=15)
model = conv_neural_func7(window_dim)
model_path = os.path.join(neural_func_dir, "models", tag + ".pth")
model.load_state_dict(torch.load(model_path))
model.eval()
output_dir = os.path.join(neural_func_dir, "inference",  tag)
os.makedirs(output_dir, exist_ok=True)
for rho in rho_list[:3]:

    c1 =  neural_c1(model, rho, batch_size=64)
    fig, ax = plt.subplots(1,2, figsize=(18, 9))
    a = ax[0].imshow(rho)
    fig.colorbar(a, ax=ax[1,0])
    b = ax[1].imshow(c1)
    ax[1].set_title("Network c1")
    fig.colorbar(b, ax=ax[0,0])
    save_path = os.path.join(output_dir,name+".png")

    plt.savefig(save_path)
    plt.close()

