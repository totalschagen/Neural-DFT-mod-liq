import torch
import numpy as np 
import pandas as pd
import os
import neural_helper as helper
data_dir = "/scratch/c7051233/Neural_functional_project/data_generation"

# Path to the Density_profiles directory
tag = "proper_2d"
density_profiles_dir = os.path.join(os.path.join(data_dir,"Density_profiles"), tag)
output_dir = os.path.join(data_dir,"training_sets")

names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]

window_stack = []
value_stack = []

for i in names[:1]:
    df,nperiod = helper.load_df(i,density_profiles_dir)
    # Get the number of rows and columns in the DataFrame
    df["muloc"]=np.log(df["rho"])+df["muloc"]
    if np.sum(df["rho"])< 0.1:
        continue
    print("proper 2d")
    print("df shape", df.shape)
    print("train length",7)
    window_dim,shape,dim,cutoff_train= helper.training_test(df, 2.5, 15,7)
    print("window_dim", window_dim, "shape", shape, "dim", dim, "cutoff_train", cutoff_train)
    print("train length",5)
    window_dim,shape,dim,cutoff_train= helper.training_test(df, 2.5, 15,5)
    print("window_dim", window_dim, "shape", shape, "dim", dim, "cutoff_train", cutoff_train)
    print("train length",10)
    window_dim,shape,dim,cutoff_train= helper.training_test(df, 2.5, 15,10)
    print("window_dim", window_dim, "shape", shape, "dim", dim, "cutoff_train", cutoff_train)


tag = "current_train"
density_profiles_dir = os.path.join(os.path.join(data_dir,"Density_profiles"), tag)
output_dir = os.path.join(data_dir,"training_sets")

names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]

window_stack = []
value_stack = []

for i in names[:1]:
    df,nperiod = helper.load_df(i,density_profiles_dir)
    # Get the number of rows and columns in the DataFrame
    df["muloc"]=np.log(df["rho"])+df["muloc"]
    if np.sum(df["rho"])< 0.1:
        continue
    print("current train")
    print("df shape", df.shape)
    print("train length",7)
    window_dim,shape,dim,cutoff_train= helper.training_test(df, 2.5, 15,7)
    print("window_dim", window_dim, "shape", shape, "dim", dim, "cutoff_train", cutoff_train)
    print("train length",5)
    window_dim,shape,dim,cutoff_train= helper.training_test(df, 2.5, 15,5)
    print("window_dim", window_dim, "shape", shape, "dim", dim, "cutoff_train", cutoff_train)
    print("train length",10)
    window_dim,shape,dim,cutoff_train= helper.training_test(df, 2.5, 15,10)
    print("window_dim", window_dim, "shape", shape, "dim", dim, "cutoff_train", cutoff_train)

