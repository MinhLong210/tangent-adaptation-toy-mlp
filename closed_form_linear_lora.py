import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import torch.nn.functional as F
import copy
from torch.func import jacrev, functional_call
from collections import OrderedDict
import time

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load dataset
batch_size = 64
trainset = torchvision.datasets.MNIST(root="../data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root="../data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# RELU with bias
# class MLP(nn.Module):
#     def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.act1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.act2 = nn.ReLU()
#         self.fc3 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten image to [batch, 784]
#         x = self.fc1(x)
#         x = self.act1(x)
#         x = self.fc2(x)
#         x = self.act1(x)
#         x = self.fc3(x)
#         return x

# Leaky ReLU with no bias
# class MLP(nn.Module):
#     def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
#         self.act1 = nn.LeakyReLU(0.1)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.act2 = nn.LeakyReLU(0.1)
#         self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten image to [batch, 784]
#         x = self.fc1(x)
#         x = self.act1(x)
#         x = self.fc2(x)
#         x = self.act1(x)
#         x = self.fc3(x)
#         return x

# LeakyReLU with bias
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image to [batch, 784]
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act1(x)
        x = self.fc3(x)
        return x

# Instantiate model
model = MLP()
model.load_state_dict(torch.load("pretrained/pretrain_mlp_LeakyRelu_bias.pth"))
model = model.to(device)

def closed_form_linear(model, train_loader, target_rank=32, slice_size=32):
    """
    Compute a closed-form-like update for a PyTorch model.
    
    Args:
        model: A pretrained model that takes inputs and returns predictions.
        train_loader: Basic train loader.
        slice_size: number of rows of each slice.
        target_rank: target low rank of each slice (the resulting cumulative rank usually = slice_size, resulting in a full-rank solution). 
    
    """
    # Initialize new model
    updated_model = copy.deepcopy(model)
    
    # Forward pass: compute predictions
    model.eval()  # Set to evaluation mode (no dropout, etc.)

    # Build param dict: keep everything the same except the one param
    global_param_dict = OrderedDict((n, p.detach()) for n, p in model.named_parameters())

    # Define function to pass to jacrev
    
    # Compute Jacobian of predictions wrt parameters
    for (name, param), (name_clone, param_clone) in zip(
        model.named_parameters(), 
        updated_model.named_parameters()
    ):  # Since LoRA is applied only to vision model
        if "fc2" in name and "bias" not in name:
            print(name)
            param.requires_grad_(True)

            # Slice params
            num_slices = param.shape[0] // slice_size
            slice_param_size = slice_size * param.shape[1]  # 16 * 768 = 12,288

            # Loop over the slices
            for slice_idx in range(num_slices):
                start_time = time.time()
                print(f"Slice: {slice_idx+1}/{num_slices}")
                slice_start = slice_idx * slice_size
                slice_end = slice_start + slice_size

                # Define the forward function for jacrev
                def model_forward(param_slice, input_images):
                    # Clone the full parameter and replace the slice
                    full_param = param.detach().clone()
                    full_param[slice_start:slice_end] = param_slice.reshape(slice_size, param.shape[1])
                    # Create a new state dict with the updated parameter
                    state_dict = {n: p.detach() if n != name else full_param for n, p in model.named_parameters()}
                    return functional_call(model, state_dict, (input_images,)).softmax(dim=1)

                # Initialize accumulated terms for each slice
                global_At_A = torch.zeros((slice_param_size, slice_param_size)).to(device)
                global_At_b = torch.zeros(slice_param_size).to(device)       
                for batch_idx, batch in enumerate(tqdm(train_loader)):
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)

                    # One-hot encode labels
                    labels = labels.squeeze()  # Shape: (n,)
                    labels = F.one_hot(labels, num_classes=10).float().to(device)  # Shape: (n, K)
                    batch_size, output_dim = labels.shape

                    with torch.no_grad():
                        logits = model(images).softmax(dim=1)
                        logits = logits.detach()
                    

                    param_slice = param[slice_start:slice_end].reshape(slice_size, param.shape[1])  # Shape: (slice_size, param.shape[1])
                    param_slice = param_slice.reshape(-1)  # Flatten for jacrev: (slice_param_size,)
                    param_slice.requires_grad_(True)

                    global_param_dict[name] = param  # This is the param we want jacrev to track

                    
                    jacobian = jacrev(model_forward)(param_slice, images)
                    # if torch.all(jacobian == 0):
                    #     print(f"Warning: Jacobian is all zeros for slice {slice_idx}, batch {batch_idx}")
                    #     # Debug: Check if output depends on param_slice
                    #     output1 = model_forward(param_slice, images)
                    #     output2 = model_forward(param_slice + 0.01, images)
                    #     print("Output change:", torch.norm(output1 - output2))
                    A_matrix_slice = jacobian.reshape(batch_size * output_dim, -1)  # Shape: (batch_size * output_dim, slice_param_size)

                    output = model_forward(param_slice, images).sum()
                    grad = torch.autograd.grad(output, param_slice)[0]
                    # print("Gradient:", grad)

                    # import pdb; pdb.set_trace()
                    # After accumulating A_matrix_slice for the batch
                    # Compute b_vector = logits - labels
                    b_vector = (logits - labels).flatten()  # Shape: (batch_size * num_classes)
                    
                    # Compute A^T A and A^T b for this batch
                    global_At_A.add_(A_matrix_slice.T @ A_matrix_slice)  # Shape: (p, p)
                    global_At_b.add_(A_matrix_slice.T @ b_vector)       # Shape: (p,)

                    
                    # Clean up
                    # Delete all intermediate tensors
                    del A_matrix_slice, b_vector, jacobian, param_slice
                    del images, labels, logits
                    # Set variables to None to ensure no references persist
                    A_matrix_slice, b_vector, jacobian, param_slice = None, None, None, None
                    images, labels, logits = None, None, None
                    # Clear the computational graph by detaching tensors (if any still require gradients)
                    if global_At_A.requires_grad:
                        global_At_A = global_At_A.detach()
                    if global_At_b.requires_grad:
                        global_At_b = global_At_b.detach()
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                
                # After processing all batches for this slice, solve the system
                # Add small regularization to ensure numerical stability
                reg_term = 1e-5 * torch.eye(global_At_A.shape[0]).to(device)
                global_At_A_reg = global_At_A + reg_term
                # import pdb; pdb.set_trace()
                
                # Compute eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(global_At_A_reg)
                
                # Sort eigenvalues and eigenvectors in descending order
                idx = eigenvalues.argsort(descending=True)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                if target_rank > 0:
                    # Project At_b onto eigenvectors
                    a_coeff = eigenvectors.T @ global_At_b
                    
                    # Selection criterion: a_coeff^2 / eigenvalues
                    selection_criterion = (a_coeff ** 2) / eigenvalues
                    
                    # Sort by selection criterion in descending order
                    sorted_indices = torch.argsort(selection_criterion, descending=True)

                    # Greedily select eigenvectors based on sorted criterion
                    cumulative_rank = 0
                    selected_indices = []

                    for idx in sorted_indices:
                        # Add the eigenvector corresponding to this index
                        selected_indices.append(idx.item())

                        # Compute temporary solution with the selected eigenvectors
                        E_t_temp = eigenvectors[:, selected_indices]
                        S_t_inv_temp = torch.diag(1.0 / eigenvalues[selected_indices])
                        temp_solution = E_t_temp @ S_t_inv_temp @ (E_t_temp.T @ global_At_b)

                        # Reshape to check rank
                        temp_matrix = temp_solution.reshape(slice_size, param.shape[1])
                        rank = torch.linalg.matrix_rank(temp_matrix)

                        cumulative_rank += rank
                        if cumulative_rank >= target_rank:
                            break
                    
                    # Compute final closed-form solution with selected components
                    E_t = eigenvectors[:, selected_indices]  # Selected eigenvectors
                    S_t_inv = torch.diag(1.0/eigenvalues[selected_indices])  # Inverse of selected eigenvalues
                    
                    # w_update = E_t @ S_t^-1 @ E_t^T @ global_At_b
                    w_update = E_t @ S_t_inv @ (E_t.T @ global_At_b)
                
                else: # full rank
                    # import pdb; pdb.set_trace()
                    w_update = eigenvectors @ torch.diag(1.0/eigenvalues) @ (eigenvectors.T @ global_At_b)
                
                # Reshape to match parameter dimensions
                w_update = w_update.reshape(slice_size, param.shape[1])
                
                # Update the parameter in the cloned model
                with torch.no_grad():
                    if param_clone.data[slice_start:slice_end].shape == w_update.shape:
                        param_clone.data[slice_start:slice_end] += w_update # If we remove w_update, the model preserves the pretrained accuracy
                    
                    else:
                        print(f"Shape mismatch: {param_clone.data[slice_start:slice_end].shape} vs {w_update.shape}")
                
                # Clean up
                del global_At_A, global_At_b, eigenvalues, eigenvectors, w_update
                torch.cuda.empty_cache()

                if target_rank > 0:
                    print(f"Slice {slice_idx+1} completed. Final rank: {cumulative_rank}")
                else:
                    print("Full rank solution")

    return updated_model




############################## Training
TARGET_RANK = 32
SLICE_SIZE = 32
updated_model = closed_form_linear(model, trainloader, TARGET_RANK, SLICE_SIZE) # Full rank closed form

updated_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = updated_model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct/total:.2f}%")
