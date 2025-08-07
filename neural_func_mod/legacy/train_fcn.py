import numpy as np 
import matplotlib.pyplot as plt
import torch
import os
import random
import fcn_network as net
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)



path = os.getcwd()
dataname = os.path.join(path, "training_data_test.pt")
data = torch.load(dataname)
inputs = data['windows']
targets = data['c1']
inputs = inputs.view(-1, 1, 85,85)  # Reshape to (N, C, H, W)
inputs = inputs.float()
targets = targets.view(-1, 1)  # Reshape to (N, 1)
dataset = torch.utils.data.TensorDataset(inputs, targets)

data_size = len(dataset)
train_size = int(0.7 * data_size)
val_size = int(0.15 * data_size)
test_size = data_size - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,worker_init_fn=worker_init_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,worker_init_fn=worker_init_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False,worker_init_fn=worker_init_fn)

model = net.fcn_neural_func()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 200

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

### Test loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

test_loss /= len(test_loader)
torch.save(model.state_dict(), "2d_fcn.pth")
print(f"Test Loss: {test_loss:.4f}")

plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("train_val_loss.png")