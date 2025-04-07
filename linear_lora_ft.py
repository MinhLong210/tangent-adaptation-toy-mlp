import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from peft.tuners.lora import LoraLayer
import torch.nn.functional as F

from LinearizedModel import LinearizedModelWraper, linearize_lora_model

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
batch_size = 64
trainset = torchvision.datasets.MNIST(root="../data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root="../data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image to [batch, 784]
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act1(x)
        x = self.fc3(x)
        return x

# Custom LoRA layer
class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA trainable parameters
        self.lora_A = nn.Parameter(torch.randn(base_layer.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(base_layer.out_features, r) * 0.01)

        # Freeze the original model weights
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_output = self.base_layer(x)  # Standard MLP forward pass
        lora_output = torch.matmul(x, self.lora_A)  # Apply LoRA A
        lora_output = torch.matmul(lora_output, self.lora_B.T) * self.scaling  # Apply LoRA B
        return base_output + lora_output  # Add LoRA adaptation

# Apply LoRA to pretrained MLP model
class LoRA_MLP(nn.Module):
    def __init__(self, pretrained_mlp, r=8, alpha=16):
        super().__init__()
        self.fc1 = pretrained_mlp.fc1  # Replace fc1
        self.act1 = nn.ReLU()
        self.fc2 = LoRALinear(pretrained_mlp.fc2, r, alpha)  # Replace fc2
        self.act2 = nn.ReLU()
        self.fc3 = pretrained_mlp.fc3  # Replace fc3
        
        # Freeze fc1 and fc3
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc3.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

# Instantiate model
pretrained_model = MLP()
pretrained_model.load_state_dict(torch.load("pretrained/pretrain_mlp.pth"))

# Apply LoRA
model = LoRA_MLP(pretrained_model, 8, 16)

# Freeze model
num_trainable = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        num_trainable += param.numel()
print("Trainable param:", num_trainable)
# Linearize lora modules
model = linearize_lora_model(model)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Training loop
epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(trainloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(trainloader):.4f}, Accuracy: {100 * correct/total:.2f}%")
    torch.save(model.state_dict(), "linear_lora_ft/linear_lora_mlp.pth")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct/total:.2f}%")
