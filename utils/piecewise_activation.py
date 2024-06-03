import torch.nn as nn
from piecewise_approximation import PieceWiseApproximation

# PieceWiseActivation
# This class implements an activation function using piecewise linear approximation.
class PieceWiseActivation(nn.Module):

    def __init__(self):
        super().__init__()
        self.piecewise = PieceWiseApproximation()

    def forward(self, input):
        # Move the input tensor to the CPU, apply operation on CPU, move back to GPU
        input_cpu = input.cpu()
        output_cpu = input_cpu.detach().apply_(self.piecewise.calculate)
        output = output_cpu.to(input.device)
        return output