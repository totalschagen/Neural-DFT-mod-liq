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
from inference.inference_func import neural_c1,neural_c2

class debug_net(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(debug_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        # get center of the input tensor
        center_x = self.input_dim[0] // 2
        center_y = self.input_dim[1] // 2
        x_center = x[ center_x, center_y]  # Assuming x is of shape (batch_size, height, width)
        tens = torch.zeros(self.output_dim, device=x.device)  # Create a tensor of zeros with the specified output dimensions
        tens[0] = x_center  # Set the first element to the center value
        return tens

data_dir, density_profiles_dir, neural_func_dir = set_paths()


data_tag = "parallel2025-06-20_00-01-10"
model_tag="20250724_020756chunk_training"

num_slices=2
output_dir = os.path.join(data_dir,"neural_func/inference","debug")
os.makedirs(output_dir, exist_ok=True)

names= helper.get_names(density_profiles_dir,data_tag,ending=".dat")
model_path = os.path.join(neural_func_dir,  model_tag , "model.pth")

# model = debug_net(input_dim=(21,21), output_dim=(1,))
model = net.conv_neural_func9(3)


rho = torch.randn(10,10)  # Example input tensor
c2 =  neural_c2(model, rho, num_slices,1,10,True)
inv_rho = 1/rho
fourier_c2 = torch.fft.rfftn(c2)
v_mu_nu = torch.fft.rfft2(inv_rho)
print(v_mu_nu.shape)
print("fourier_c2 shape", fourier_c2.shape)

fig, ax = plt.subplots(1,2, figsize=(18, 9))
a = ax[0].imshow(rho)
fig.colorbar(a, ax=ax[0])
b = ax[1].imshow(c1)
ax[1].set_title("Network c1")
fig.colorbar(b, ax=ax[1])
save_path = os.path.join(output_dir,"test.png")

plt.savefig(save_path)
plt.close()

