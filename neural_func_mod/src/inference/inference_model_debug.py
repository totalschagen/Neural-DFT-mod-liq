print("starting")
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

model_tag = str(sys.argv[1]) if len(sys.argv) > 1 else "20250730_033617chunk_training"
model_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 401
reverse = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

print(f"Reverse: {reverse}")
timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(data_dir,"neural_func/picard",timestamp)
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(neural_func_dir,  model_tag , "best_model.pth")

model = net.conv_neural_func5(model_dim,8)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
device= torch.device("cuda")
model = model.to(device) 
_,_, window_dim,_ = next(model.children()).weight.shape
print(f"Model loaded from {model_path} with window dimension {window_dim}")
L = 10
T = 1
nperiod=[10,20]
mu=[0.5,4.0,8.0]
amp = [0.1,0.5]
if reverse:
    nperiod = nperiod[::-1]
    mu = mu[::-1]
    amp = amp[::-1]
for n in nperiod:
    for m in mu:
        for am in amp:
            if n==10 and m==0.5:
                continue
            print(f"Calculating for nperiod={n}, mu={m}, amp={am}")
            rho,c1 = picard_minimization(L=L,mu=m,T=T,model=model,nperiod=n,Amp=am)
            np.save(os.path.join(output_dir,f"rho_nperiod{n}_mu{m}_amp{am}.npy"),rho)
            np.save(os.path.join(output_dir,f"c1_nperiod{n}_mu{m}_amp{am}.npy"),rho)
            fig, ax = plt.subplots(1,1, figsize=(18, 6))
            a = ax.imshow(rho)
            fig.colorbar(a, ax=ax)
            save_path = os.path.join(output_dir,f"rho_n{n}_mu{m}_amp{am}.png")
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
            plt.close()


