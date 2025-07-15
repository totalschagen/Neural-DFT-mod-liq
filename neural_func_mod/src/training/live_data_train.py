import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
import random
import sys
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import models.conv_network as net
import utils.neural_helper as helper
from utils.set_paths import set_paths
from data_pipeline.create_training_data_func import load_training_data
from data_pipeline.data_loader_class import ChunkManager, SlidingWindowDataset

def get_model_param_memory_mb(model):
    total_bytes = sum(p.element_size() * p.nelement() for p in model.parameters())
    return total_bytes / 1024**2

def get_gradient_memory_mb(model):
    total_bytes = 0
    for param in model.parameters():
        if param.grad is not None:
            total_bytes += param.grad.element_size() * param.grad.nelement()
    return total_bytes / 1024**2  # Convert to MB



if len(sys.argv) < 3:
    print("please job name and data tag")
    sys.exit(1)

job = str(sys.argv[1])
tag = str(sys.argv[2])
window_L = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
num_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 200
data_dir, density_profiles_dir, neural_func_dir = set_paths()

## Prepare data paths
timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(neural_func_dir,timestamp+tag)
fig_output_path= os.path.join(output_dir,"training_metrics")
diagnostic_output_path = os.path.join(output_dir,"RAM_usage")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_output_path, exist_ok=True)
os.makedirs(diagnostic_output_path, exist_ok=True)

### load and prepare data set
rho_list,c1_list,window_dim,dx= load_training_data(tag,window_L=window_L,sim_L=15)

n_rho = len(rho_list)
print(n_rho, window_dim, dx)
# ---- Step 2: Create a list dataset (matrix-level granularity) ----
all_data = list(zip(rho_list, c1_list))

# ---- Step 3: Split into train/val/test sets ----
lengths = [
    int(0.7 * n_rho),  # 70%
    n_rho     - int(0.7 * n_rho) 
]

"""
generator = torch.Generator().manual_seed(42)
train_data, val_data = random_split(all_data, lengths, generator=generator)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")
# ---- Step 4: Build datasets from each split ----
def build_dataset(data, window_size):
    inputs, labels = zip(*data)
    return SlidingWindowDataset(inputs, labels, window_size)
# ---- Step 3: Create datasets ----
print("Building datasets...")
train_dataset = build_dataset(train_data, window_dim)
val_dataset   = build_dataset(val_data, window_dim)

print("Wrapping datasets in DataLoaders...")
# ---- Step 4: Wrap in DataLoaders ----
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)#, num_workers=5, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False)#, num_workers=5, pin_memory=True)

helper.get_mem_usage(diagnostic_output_path)
## prepare model
model = net.conv_neural_func7(window_dim)
#model.load_state_dict(torch.load("2d_conv.pth"))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 200

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
device = torch.device("cuda")
print("Using devicee:",device)
model.to(device)



train_loss = []
validation_loss = []
print("Commencing training...")
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    ### Training loop
    for inputs,targets in train_loader:
        #print("Shifting inputs and targets to GPU")
        # inputs = inputs.to(device)
        # targets = targets.to(device)
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
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs).view(-1)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    validation_loss.append(val_loss)
    print(f"Validation Loss: {val_loss:.4f}")
torch.cuda.empty_cache()
"""
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

# print("Wrapping datasets in DataLoaders...")
# # ---- Step 4: Wrap in DataLoaders ----
# train_loader = DataLoader(train_dataset, batch_size=400, shuffle=True, num_workers=0)#, pin_memory=True)
# val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)#, pin_memory=True)
# # test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

#helper.get_mem_usage(diagnostic_output_path)
## prepare model
model = net.conv_neural_func7(window_dim)
#model.load_state_dict(torch.load("2d_conv.pth"))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#num_epochs = 200

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
device = torch.device("cuda")
print("Using devicee:",device)
model.to(device)


torch.cuda.memory._record_memory_history(max_entries=100000)
train_loss = []
validation_loss = []
train_manager = ChunkManager(train_inputs, train_labels, window_dim,stride=6, chunk_size=3, batch_size=128, shuffle=True, pin_memory=True, num_workers=0)
val_manager = ChunkManager(val_inputs, val_labels, window_dim, stride=5,chunk_size=1, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)
print("Commencing training...")
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    for chunk_idx in range(len(train_manager)):
        train_loader = train_manager.get_chunk_loader(chunk_idx)
        chunk_running_loss = 0.0
        ### Training loop
        for inputs,targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            chunk_running_loss += loss.item()
        chunk_running_loss /= len(train_loader)
        running_loss += chunk_running_loss
        
    running_loss /= len(train_manager)
    train_loss.append(running_loss)
    torch.cuda.empty_cache()
        # if epoch > 300 and epoch % 50 == 0:
        #     scheduler.step()
    print(f"Learning rate: {scheduler.get_last_lr()[0]}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")
    print(f"Param memory: {get_model_param_memory_mb(model):.2f} MB")
    print(f"Grad memory:  {get_gradient_memory_mb(model):.2f} MB")
    model.eval()
    running_loss = 0.0
    for chunk_idx in range(len(val_manager)):    
        val_loader = val_manager.get_chunk_loader(chunk_idx)
        chunk_running_loss = 0.0
        ### Validation loop
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs).view(-1)

                loss = criterion(outputs, targets)
                chunk_running_loss += loss.item()
        chunk_running_loss /= len(val_loader)
        running_loss += chunk_running_loss
    running_loss /= len(val_manager)
    validation_loss.append(running_loss)
    print(f"Validation Loss: {running_loss:.4f}")

    torch.cuda.empty_cache()
        
try:
    torch.cuda.memory._dump_snapshot(f"{os.path.join(diagnostic_output_path,"vram_snapshot")}.pickle")
except Exception as e:
    logger.error(f"Failed to capture memory snapshot {e}")
torch.cuda.memory._record_memory_history(enabled=None)
torch.cuda.empty_cache()


plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(fig_output_path,"train_val_loss_conv"+".png"))

torch.save(model.state_dict(),output_dir+"/model.pth")