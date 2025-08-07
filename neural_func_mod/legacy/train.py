import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
import random
import models.conv_network as net
import utils.neural_helper as helper
from utils.set_paths import neural_func_dir

## Prepare data paths
training_data_path = os.path.join(neural_func_dir,"training_sets")
dataname = os.path.join(training_data_path, "check_train.pt")
output_path= os.path.join(neural_func_dir,"training_metrics")

### load and prepare data set
data = torch.load(dataname)
inputs = data['windows']
targets = data['c1']
inputs = inputs.view(-1, 1, 213,213)  # Reshape to (N, C, H, W)
inputs = inputs.float()
targets = targets.view(-1,1,1 ,1)  # Reshape to (N, 1)
dataset = torch.utils.data.TensorDataset(inputs, targets)

## split data
data_size = len(dataset)
train_size = int(0.7 * data_size)
val_size = int(0.15 * data_size)
test_size = data_size - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,worker_init_fn=worker_init_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,worker_init_fn=worker_init_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False,worker_init_fn=worker_init_fn)

## prepare model
model = net.conv_neural_func7()
#model.load_state_dict(torch.load("2d_conv.pth"))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 200

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
device = torch.device("cuda")
model.to(device)



train_loss = []
validation_loss = []
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    ### Training loop
    for inputs,targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        ## update weights
        #print(loss.item())
        optimizer.step()
        running_loss += loss.item()
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
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    validation_loss.append(val_loss)
    print(f"Validation Loss: {val_loss:.4f}")
torch.cuda.empty_cache()



torch.save(model.state_dict(), "2d_conv.pth")

plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(output_path,"train_val_loss_conv"+".png"))
