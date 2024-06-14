import torch
import torch.nn.functional as F

def torch_correlate2d(input, kernel, padding='same'):
    # Convert input and kernel to tensors if they are not already
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input, dtype=torch.float32)
    if not isinstance(kernel, torch.Tensor):
        kernel = torch.tensor(kernel, dtype=torch.float32)
    
    # Ensure the input is 4D (batch_size, channels, height, width)
    if input.dim() == 2:
        input = input.unsqueeze(0).unsqueeze(0)
    elif input.dim() == 3:
        input = input.unsqueeze(0)
    
    # Ensure the kernel is 4D (out_channels, in_channels, height, width)
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)
    elif kernel.dim() == 3:
        kernel = kernel.unsqueeze(0)

    # Flip the kernel horizontally and vertically
    kernel = torch.flip(kernel, [2, 3])
    
    # Calculate padding
    if padding == 'same':
        pad = (kernel.size(2) // 2, kernel.size(3) // 2)
    else:
        pad = (0, 0)

    # Perform correlation using conv2d
    result = F.conv2d(input, kernel, padding=pad)

    # Remove the extra dimensions if input was 2D
    if result.size(0) == 1:
        result = result.squeeze(0)
    if result.size(0) == 1:
        result = result.squeeze(0)
    
    return result