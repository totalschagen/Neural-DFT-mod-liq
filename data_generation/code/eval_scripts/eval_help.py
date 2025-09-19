import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib import patches as mpatches
import pandas as pd
import os
import math

def load_df(name,directory):
    name = os.path.join(directory, name)
    print("Reading file: ", name)
    df = pd.read_csv(name,delimiter = " ")

    extra=[]
    try:
        nperiod = (list(df.columns)[4])
        df.drop(columns=[nperiod],inplace=True)
        nperiod = int(nperiod)
        extra.append(nperiod)
    except:
        print("No nperiod")
    try:
        mu = (list(df.columns)[4])

        df.drop(columns=[mu],inplace=True)
        mu = float(mu)
        extra.append(mu)
    except:
        print("No mu")
    try:
        packing = (list(df.columns)[4])
        df.drop(columns=[packing],inplace=True)
        packing = float(packing)
        extra.append(packing)
    except:
        print("No packing")
    try:
        packing = (list(df.columns)[4])
        df.drop(columns=[packing],inplace=True)
        packing = float(packing)
        extra.append(packing)
    except:
        print("No amp")
    extra_dic = {"nperiod": extra[0], "mu": extra[1], "packing": extra[2], "amp": extra[3]}
    return df, extra_dic


def plot_density_profiles_2d_3d(tag,num=-1,start=0):
    parent_dir = "/home/c705/c7051233/scratch_neural/data_generation"
    # Path to the Density_profiles directory
    density_profiles_dir = os.path.join(parent_dir, "Density_profiles",tag)
    plotname = tag + "dens_plot.png"
    savename = os.path.join(parent_dir,"density_plots",plotname)
    names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]
    slice = names[start:num]
    rows =len(slice)
    cols = 1
    ## create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(9,9*rows))
    fig.suptitle("2D Density profile")
    
    for i, name in enumerate(slice):
        # Get the current dataframe
        df, extra = load_df(name, density_profiles_dir)
        print(len(df))
        label = f"Nperiod = {extra["nperiod"]}, mu = {extra["mu"]}, packing = {extra["packing"]}, amp = {extra["amp"]}"
        shape = np.sqrt(len(df)).astype(int)
        xi = df["x"].values.reshape(shape, shape)
        yi = df["y"].values.reshape(shape,shape)
        rho = df["rho"].values.reshape(shape,shape)
        pcm = axs[i].pcolormesh(xi, yi, rho/2, shading='auto', cmap='viridis',label=label)
        axs[i].set_xlabel(r"$x/\sigma$")
        axs[i].set_ylabel(r"$y/\sigma$")
        patch = mpatches.Patch(color=pcm.cmap(0.5), label=label)
        axs[i].legend(handles=[patch])

        # x_values = df.groupby("x").mean().reset_index()["x"]
        # rho_values = df.groupby("x").mean().reset_index()["rho"]

        # axs[i,1].plot(x_values, rho_values, label="rho")
        # axs[i,1].set_title(slice[i])
        # axs[i,1].set_xlabel(r"$x/\sigma$")
        # axs[i,1].set_ylabel(r"$\rho$")



        fig.colorbar(pcm, ax=axs[i], label=r"$\rho^{(1)}(\mathbf{x})$") 
    plt.tight_layout()
    plt.savefig(savename, dpi=300)

def plot_density_profiles_2d_3d_debug(tag,num=-1,start=0):
    parent_dir = "/home/c705/c7051233/scratch_neural/data_generation"
    # Path to the Density_profiles directory
    density_profiles_dir = os.path.join(parent_dir, "Density_profiles",tag)
    plotname = tag + "dens_plot.png"
    savename = os.path.join(parent_dir,"density_plots",plotname)
    names= [f for f in os.listdir(density_profiles_dir) if os.path.isfile(os.path.join(density_profiles_dir, f))]
    slice = names[start:num]
    rows =len(slice)
    cols = 2
    ## create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(11,10*rows/2))
    fig.suptitle("2D Density profile")
    
    for i, name in enumerate(slice):
        # Get the current dataframe
        df, extra = load_df(name, density_profiles_dir)
        print(len(df))
        label = f"Nperiod = {extra["nperiod"]}, mu = {extra["mu"]}, packing = {extra["packing"]}, amp = {extra["amp"]}"
        shape = np.sqrt(len(df)).astype(int)
        xi = df["x"].values.reshape(shape, shape)
        yi = df["y"].values.reshape(shape,shape)
        rho = df["rho"].values.reshape(shape,shape)
        pcm = axs[i,0].pcolormesh(xi, yi, rho/2, shading='auto', cmap='viridis',label=label)
        axs[i,0].set_xlabel(r"$x/\sigma$")
        axs[i,0].set_ylabel(r"$y/\sigma$")
        patch = mpatches.Patch(color=pcm.cmap(0.5), label=label)
        axs[i,0].legend(handles=[patch])

        x_values = df.groupby("x").mean().reset_index()["x"]
        rho_values = df.groupby("x").mean().reset_index()["rho"]

        axs[i,1].plot(x_values, rho_values, label="rho")
        axs[i,1].set_title(slice[i])
        axs[i,1].set_xlabel(r"$x/\sigma$")
        axs[i,1].set_ylabel(r"$\rho$")



        fig.colorbar(pcm, ax=axs[i,0], label=r"$\rho^{(1)}(\mathbf{x})$") 
    plt.tight_layout()
    plt.savefig(savename, dpi=300)