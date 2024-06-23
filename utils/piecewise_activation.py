import torch.nn as nn

# This class implements an activation function using piecewise linear approximation.
class PiecewiseActivation(nn.Module):
    
    def __init__(self, gelu_piecewise_approx):
        super().__init__()
        self.piecewise = gelu_piecewise_approx

    def forward(self, input):
        # Move the input tensor to the CPU, apply operation on CPU, move back to GPU
        input_cpu = input.cpu()
        output_cpu = input_cpu.detach().apply_(self.piecewise.calculate)
        output = output_cpu.to(input.device)
        return output