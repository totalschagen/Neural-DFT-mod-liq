import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
import sys
from datetime import datetime
import models.conv_network as net
import utils.neural_helper as helper
from utils.set_paths import set_paths
from data_pipeline.create_training_data_func import load_training_data
from inference.inference_func import picard_minimization


data_dir, density_profiles_dir, neural_func_dir = set_paths()
picard_tag = "20250913_001536"
gcmc_tag="parallel2025-09-11_18-37-34"
picard_dir = os.path.join(data_dir,"neural_func/picard",picard_tag)
rho_pic = np.load(os.path.join(picard_dir,"rho_nperiod10_mu0.5_amp0.5.npy"))
c1_pic = np.load(os.path.join(picard_dir,"c1_nperiod10_mu0.5_amp0.5.npy"))
gcmc_dir = os.path.join(density_profiles_dir,gcmc_tag)
df,extra = helper.load_df(os.path.join(gcmc_dir,"rho_MC_2D_3.dat"))
print(extra)
rhomatrix = df.pivot(index='y', columns='x', values='rho').values
picard_shape = rho_pic.shape
rho_gcmc = rhomatrix[:picard_shape[0],:picard_shape[1]]
fig, ax = plt.subplots(2,2, figsize=(14, 14))
a = ax[0,0].imshow(rho_pic,vmin=0.15,vmax=0.7)
fig.colorbar(a, ax=ax[0,0], label=r"$\rho^{(1)}(\mathbf{x})$",fraction=0.047*0.5)

ax[0,0].set_title("Neural Functional")

ax[0,0].set_xlabel(r"$L_x = 10\sigma$")
ax[0,0].set_ylabel(r"$L_y = 5\sigma$")

b = ax[0,1].imshow(rho_gcmc,vmin=0.15,vmax=0.7)
fig.colorbar(b, ax=ax[0,1], label=r"$\rho^{(1)}(\mathbf{x})$",fraction=0.047*0.5)
ax[0,1].set_title("GCMC Simulation")
ax[0,1].set_xlabel(r"$L_x = 10\sigma$")
ax[0,1].set_ylabel(r"$L_y = 5\sigma$")
ax[1,0].set_xlabel(r"$L_x = 10\sigma$")
ax[1,0].set_ylabel(r"$L_y = 5\sigma$")
ax[1,1].set_xlabel(r"$L_x = 10\sigma$")
ax[1,1].set_ylabel(r"$L_y = 5\sigma$")
c = ax[1,0].imshow((rho_pic - rho_gcmc)/10)
fig.colorbar(c, ax=ax[1,0], label=r"$\rho_{NF}^{(1)}(\mathbf{x}) - \rho_{GCMC}^{(1)}(\mathbf{x})$",fraction=0.047*0.5)
ax[1,0].set_title("Difference")
d = ax[1,1].imshow(c1_pic)
fig.colorbar(d, ax=ax[1,1], label=r"$c^{(1)}(\mathbf{x})$",fraction=0.047*0.5)
ax[1,1].set_title("Direct Correlation Function from NF")
plt.tight_layout()
save_path = os.path.join(data_dir,"neural_func/picard",picard_tag,"comparison_picard_gcmc.png")
plt.savefig(save_path)
print(f"Figure saved to {save_path}")