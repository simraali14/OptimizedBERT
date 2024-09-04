import torch
import torch.nn as nn

class invSqrt(nn.Module):
    def __init__(self, piecewise_approx):
        super().__init__()
        self.piecewise_approx = piecewise_approx

    def forward(self, input):
        input_cpu = input.cpu()
        output_cpu = input_cpu.detach().apply_(self.piecewise_approx.calculate)
        output = output_cpu.to(input.device)
        return output

class PiecewiseLayerNorm(nn.Module):
      def __init__(self, weight, bias, layernorm_piecewise_approx):
        super().__init__()
        self.weight  = weight
        self.bias    = bias
        self.invsqrt = invSqrt(layernorm_piecewise_approx)

      def forward(self, input):
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = torch.square(input - mean).mean(dim = -1, keepdim=True)
        unaffined =  (input - mean) * self.invsqrt(var + .00001) #.00001 is epsilon, should change that
        return unaffined*self.weight + self.bias