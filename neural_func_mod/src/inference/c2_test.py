import numpy as np
import torch
import os
import conv_network as net
import matplotlib.pyplot as plt
import neural_helper as helper
import pandas

model = net.conv_neural_func7()
model_name = "2d_conv"
model.load_state_dict(torch.load(os.path.join("Model_weights",model_name+".pth")))
model.to(torch.device("cuda"))

parent_dir = os.path.dirname(os.getcwd())
# Path to the Density_profiles directory
density_profiles_dir =  "/home/hagen/Documents/master_project/Project/Data_generation/Density_profiles"
tag = "current_train"
density_profiles_dir = os.path.join(density_profiles_dir, tag)

output_dir_name = "inference_out"
output_dir = os.path.join(os.getcwd(),os.path.join( output_dir_name,model_name))
os.makedirs(output_dir, exist_ok=True)


names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]
name = names[4]
df,_ = helper.load_df(name,density_profiles_dir)
rhomatrix = df.pivot(index='y', columns='x', values='rho').values
rhomatrix = rhomatrix[:int(np.sqrt(len(df))/10),:int(np.sqrt(len(df))/10)]


c1_reconstruct = helper.neural_c2(model,rhomatrix)
