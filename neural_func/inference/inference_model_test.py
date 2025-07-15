import numpy as np
import torch
import os
import conv_network as net
import matplotlib.pyplot as plt
import neural_helper as helper
import pandas
datapath = "/home/c705/c7051233/scratch_neural"
model = net.conv_neural_func7()
model_name = "2d_conv"
model.load_state_dict(torch.load(os.path.join(datapath,"neural_func/models",model_name+".pth")))
model.to(torch.device("cuda"))

# Path to the Density_profiles directory
density_profiles_dir =  os.path.join(datapath, "datasets/density_profiles")
tag = "current_train"
density_profiles_dir = os.path.join(density_profiles_dir, tag)

output_dir_name = "neural_func/inference_out"
output_dir = os.path.join( datapath,output_dir_name,model_name)
os.makedirs(output_dir, exist_ok=True)


names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]
for name in names[10:30]:
    df,_ = helper.load_df(name,density_profiles_dir)
    rhomatrix = df.pivot(index='y', columns='x', values='rho').values
    df["muloc"]=np.log(df["rho"])+df["muloc"]
    mulocmatrix = df.pivot(index='y', columns='x', values='muloc').values
    rhomatrix = rhomatrix[:int(np.sqrt(len(df))/12),:int(np.sqrt(len(df))/12)]
    mulocmatrix = mulocmatrix[:int(np.sqrt(len(df))/12),:int(np.sqrt(len(df))/12)]

    
    c1_reconstruct = helper.neural_c1(model,rhomatrix)
    cutout_reconstruct = mulocmatrix
    fig, ax = plt.subplots(2,2, figsize=(18, 6))
    fig.suptitle(name)
    a = ax[1,0].imshow(c1_reconstruct)
    fig.colorbar(a, ax=ax[1,0])
    ax[1,0].set_title("Network c1")
    b = ax[0,0].imshow(mulocmatrix)
    fig.colorbar(b, ax=ax[0,0])
    ax[0,0].set_title("Original c1")
    c = ax[0,1].imshow(cutout_reconstruct)
    fig.colorbar(c, ax=ax[0,1])
    ax[0,1].set_title("Cutout original c1")
    d = ax[1,1].imshow(np.array(c1_reconstruct)-np.array(mulocmatrix))
    fig.colorbar(d, ax=ax[1,1])
    ax[1,1].set_title("difference")
    save_path = os.path.join(output_dir,name+".png")

    plt.savefig(save_path)
    plt.close()

