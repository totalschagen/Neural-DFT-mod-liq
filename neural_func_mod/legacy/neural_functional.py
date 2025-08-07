import neural_helper as helper
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import conv_network as net

model = net.conv_neural_func7()
model_name = "2d_conv"
model.load_state_dict(torch.load(os.path.join("Model_weights",model_name+".pth")))
model.to(torch.device("cuda"))

rho,X,Y = helper.picard_minimization(15,0.583,1,0.05,model,20,3,alpha=0.1)

plt.figure()
plt.pcolormesh(X,Y,rho,shading='auto',cmap='viridis')
plt.colorbar()
plt.title("Density")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.savefig("picard_density.png")
plt.close()
