import torch
import numpy as np 
import pandas as pd
import os
import neural_helper as helper
data_dir = "/home/c705/c7051233/scratch_neural/"

# Path to the Density_profiles directory
tag = "proper_2d"
density_profiles_dir = os.path.join(os.path.join(data_dir,"data_generation/Density_profiles"), tag)
output_dir = os.path.join(data_dir,"training_sets")

names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]

window_stack = []
value_stack = []

for i in names[:]:
    df,extra = helper.load_df(i,density_profiles_dir)
    # Get the number of rows and columns in the DataFrame
    L_period_perturb = 15/extra["period_perturb"]
    df["muloc"] = df["muloc"]+ extra["amp_perturb"]* np.sin(2 * np.pi * df["y"] / L_period_perturb)
    df.loc[df["rho"] <= 0, "rho"] = 1e-10
    df["muloc"]=np.log(df["rho"])+df["muloc"]
    if np.sum(df["rho"])< 0.1:
        continue
    window,value= helper.build_training_data_torch_optimized(df, 2, 15,5)
    window_stack.append(window)
    value_stack.append(value)
window_tensor = window_stack[0]
value_tensor = value_stack[0]
for i in range(1,len(window_stack)):
    window_tensor = torch.cat((window_tensor, window_stack[i]), 0)
    value_tensor = torch.cat((value_tensor, value_stack[i]), 0)
print(window_tensor.shape, value_tensor.shape)

torch.save({"windows": window_tensor, "c1": value_tensor}, os.path.join(output_dir,tag+".pt"))


