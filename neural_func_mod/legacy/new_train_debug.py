import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
import random
from torch.utils.data import DataLoader, random_split
import models.conv_network as net
import utils.neural_helper as helper
from utils.set_paths import set_paths
from data_pipeline.create_training_data_func import load_training_data
from data_pipeline.data_loader_class import ChunkManager, SlidingWindowDataset

data_dir, density_profiles_dir, neural_func_dir = set_paths()
tag = "debug_set"
## Prepare data paths
fig_output_path= os.path.join(neural_func_dir,"training_metrics")
model_output_path= os.path.join(neural_func_dir,"models",tag+".pth")
process_id = os.getpid()
diagnostic_output_path = os.path.join(neural_func_dir,"RAM_usage")
os.makedirs(fig_output_path, exist_ok=True)
os.makedirs(model_output_path, exist_ok=True)
os.makedirs(diagnostic_output_path, exist_ok=True)
diagnostic_output_path = os.path.join(diagnostic_output_path, f"RAM_usage_{process_id}.txt")

### load and prepare data set
rho_list,c1_list,window_dim,dx= load_training_data(tag,window_L=0.5,sim_L=15,n_profiles=3)

n_rho = len(rho_list)
print(n_rho, window_dim, dx)
# ---- Step 2: Create a list dataset (matrix-level granularity) ----
all_data = list(zip(rho_list, c1_list))

# ---- Step 3: Split into train/val/test sets ----
lengths = [
    int(0.7 * n_rho),  # 70%
    n_rho     - int(0.7 * n_rho) 
]

generator = torch.Generator().manual_seed(42)
train_data, val_data = random_split(all_data, lengths, generator=generator)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")
# # ---- Step 4: Build datasets from each split ----
# def build_dataset(data, window_size):
#     inputs, labels = zip(*data)
#     return SlidingWindowDataset(inputs, labels, window_size)
# # ---- Step 3: Create datasets ----
# print("Building datasets...")
# train_dataset = build_dataset(train_data, window_dim)
# val_dataset   = build_dataset(val_data, window_dim)
# # test_dataset  = build_dataset(test_data, window_dim)
train_inputs, train_labels = zip(*train_data)
val_inputs, val_labels = zip(*val_data)
train_manager = ChunkManager(train_inputs, train_labels, window_dim, chunk_size=5, batch_size=200, shuffle=True, pin_memory=True, num_workers=0)
val_manager = ChunkManager(val_inputs, val_labels, window_dim, chunk_size=5, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)
# print("Wrapping datasets in DataLoaders...")
# # ---- Step 4: Wrap in DataLoaders ----
# train_loader = DataLoader(train_dataset, batch_size=400, shuffle=True, num_workers=0)#, pin_memory=True)
# val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)#, pin_memory=True)
# # test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

helper.get_mem_usage(diagnostic_output_path)
## prepare model
model = net.conv_neural_func7(window_dim)
#model.load_state_dict(torch.load("2d_conv.pth"))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 200

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
device = torch.device("cuda")
print("Using devicee:",device)
model.to(device)



train_loss = []
validation_loss = []
print("Commencing training...")
for epoch in range(num_epochs):
    for chunk_idx in range(len(train_manager)):
        train_loader = train_manager.get_chunk_loader(chunk_idx)
        val_loader = val_manager.get_chunk_loader(chunk_idx)
        running_loss = 0.0
        model.train()
        ### Training loop
        for inputs,targets in train_loader:
            #print("Shifting inputs and targets to GPU")
        # inputs = inputs.to(device)
        # targets = targets.to(device)
            #print(inputs.device)
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            ## update weights
            #print(loss.item())
            optimizer.step()
            running_loss += loss.item()
            
        helper.get_mem_usage(diagnostic_output_path)
        running_loss /= len(train_loader)
        train_loss.append(running_loss)
        if epoch > 300 and epoch % 50 == 0:
            scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

        ### Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                #inputs = inputs.to(device)
                #targets = targets.to(device)
                outputs = model(inputs).view(-1)

                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        validation_loss.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
torch.cuda.empty_cache()



torch.save(model.state_dict(),model_output_path)

plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(fig_output_path,"train_val_loss_conv"+".png"))
