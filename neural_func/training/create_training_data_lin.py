import numpy as np 
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.getcwd())
# Path to the Density_profiles directory
density_profiles_dir =  "/home/hagen/Documents/master_project/Data_generation/Density_profiles"
tag = "parallel2025-05-09_18-33-17"
density_profiles_dir = os.path.join(density_profiles_dir, tag)


def load_df(num):
    name = os.path.join( density_profiles_dir, num)
    df = pd.read_csv(name,delimiter = " ")
    nperiod = (list(df.columns)[-2])
    mu = (list(df.columns)[-1])
    df.drop(columns=[mu],inplace=True)
    df.drop(columns=[nperiod],inplace=True)
    df = df.groupby("x").mean()
    df = df.reset_index()
    nperiod = int(nperiod)
    return df,nperiod


def cut_density_intervals(df, width,L,window_stack,center_values):
    n_windows = int(L/width)
    window_dim = int((len(df))/n_windows)
    print(window_dim,len(df))
    rho = df["rho"].values
    muloc = df["muloc"].values
    for i in range(n_windows):
        window = rho[i*window_dim:(i+1)*window_dim]
        window_stack.append(window)
        center_x_index = int((i+0.5)*window_dim)
        center = muloc[center_x_index]
        center_values.append(center)
        #print(len(window))
    return window_stack,center_values

def save_matrices(inputs,targets):
    inputs = torch.tensor(inputs,dtype=torch.float32)
    targets = torch.tensor(targets,dtype=torch.float32)
    torch.save({"windows": inputs, "c1": targets}, "training_data_test_1d.pt")
names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]

window_stack = []
value_stack = []

for i in names:
    df,nperiod = load_df(i)
    # Get the number of rows and columns in the DataFrame
    df["muloc"]=np.log(df["rho"])+df["muloc"]
    if np.sum(df["rho"])< 0.1:
        continue
    window_stack,value_stack= cut_density_intervals(df, 2.5, 15,window_stack,value_stack)

window_stack = np.array(window_stack)
value_stack = np.array(value_stack)
print(window_stack.shape, value_stack.shape)
window_stack = np.stack(window_stack)
value_stack = np.stack(value_stack)
print(window_stack.shape, value_stack.shape)


save_matrices(window_stack,value_stack)
