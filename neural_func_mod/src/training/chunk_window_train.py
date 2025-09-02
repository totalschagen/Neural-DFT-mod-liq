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
from data_pipeline.create_training_data_func import load_data_chunk
from data_pipeline.data_loader_class import prepared_windows_dataset,prepared_windows_shuffler

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
num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 200
num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
cache_size = int(sys.argv[5]) if len(sys.argv) > 5 else 2
pre_train = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
batch_size = int(sys.argv[7]) if len(sys.argv) > 7 else 1024
learning_rate = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0035
model_ind = int(sys.argv[9]) if len(sys.argv) > 9 else 2
kmax = int(sys.argv[10]) if len(sys.argv) > 10 else 12

data_dir, density_profiles_dir, neural_func_dir = set_paths()

model_tag = "low_res_prov_1"


print(f"Starting training with {num_epochs} epochs, {num_workers} workers, {cache_size} cache size, pre_train={pre_train}, batch_size={batch_size}, learning_rate={learning_rate}, model_ind={model_ind}, kmax={kmax}")
## Prepare data paths
timestamp =datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(neural_func_dir,timestamp+"chunk_training")
fig_output_path= os.path.join(output_dir,"training_metrics")
diagnostic_output_path = os.path.join(output_dir,"RAM_usage")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_output_path, exist_ok=True)
os.makedirs(diagnostic_output_path, exist_ok=True)
dataset_dir = os.path.join(data_dir,"datasets")
model_path = os.path.join(neural_func_dir, model_tag, "model.pth")
### load and prepare data set
#data_list,window_dim= load_data_chunk(dataset_dir)
len_dic = helper.parse_len_file_to_dict(os.path.join(dataset_dir,tag,"size.txt"))
names = helper.get_names(dataset_dir,tag)

n_data = len(names)

lengths = [
    int(0.7 * n_data),  # 70%
    n_data     - int(0.7 * n_data) 
]


generator = torch.Generator().manual_seed(42)
train_data, val_data = random_split(names, lengths, generator=generator)
#print(f"Train: {len(train_data)}, Val: {len(val_data)}")
check_data = torch.load(names[0])
window_dim = check_data['windows'][0].shape[1]
print(f"Window dimension: {window_dim}")
model_function_list=[net.conv_neural_func3, net.conv_neural_func5, net.conv_neural_func7]
model = model_function_list[model_ind](window_size=window_dim, k=kmax)
# if model_ind == 0:
#     model = net.conv_neural_func3(window_size=window_dim, k=kmax)
# elif model_ind == 1:
#     model = net.conv_neural_func5(window_size=window_dim, k=kmax)
# elif model_ind == 2:
#     model = net.conv_neural_func7(window_size=window_dim, k=kmax)

if pre_train:
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load(model_path))
    print("Pre-trained model loaded.")
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
device = torch.device("cuda")
print("Using device:",device)
model.to(device)


torch.cuda.memory._record_memory_history(max_entries=100000)
train_loss = []
validation_loss = []
#train_manager = window_chunk_manager(train_data,batch_size=512,shuffle=True)
#val_manager = window_chunk_manager(val_data,batch_size=512,shuffle=False)
train_dataset = prepared_windows_dataset(train_data,len_dic,cache_size=cache_size)
val_dataset = prepared_windows_dataset(val_data,len_dic,cache_size=cache_size)
sampler = prepared_windows_shuffler(train_dataset)
print("Initializing DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=1024, sampler=sampler, num_workers=num_workers, pin_memory=True,prefetch_factor=4)
val_loader = DataLoader(val_dataset, batch_size=1024,  num_workers=num_workers, pin_memory=True,prefetch_factor=4)
print("Commencing training...")
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    ### Training loop
    for inputs,targets in train_loader:
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs).view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        running_loss += loss.item()
    running_loss /= len(train_loader)
        
    train_loss.append(running_loss)
    if epoch > 50 and epoch % 10 == 0:
        scheduler.step()
    print(f"Learning rate: {scheduler.get_last_lr()[0]}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")
    #print(f"Param memory: {get_model_param_memory_mb(model):.2f} MB")
    #print(f"Grad memory:  {get_gradient_memory_mb(model):.2f} MB")
    model.eval()
    running_loss = 0.0
    ### Validation loop
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    running_loss /= len(val_loader)
    validation_loss.append(running_loss)
    print(f"Validation Loss: {running_loss:.4f}")
    # if running_loss < 1.53:
    #     torch.save(model.state_dict(),output_dir+"/best_model.pth")
    #     print(f"Best model saved to {output_dir}/best_model.pth at epoch {epoch+1} with validation loss {running_loss:.4f}")
    
        
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
print(timestamp+"chunk_training")
print(f"Model saved to {output_dir}/model.pth")
print(f"Running training with {num_epochs} epochs, {num_workers} workers, {cache_size} cache size, pre_train={pre_train}, batch_size={batch_size}, learning_rate={learning_rate}, model_ind={model_ind}, kmax={kmax}")
