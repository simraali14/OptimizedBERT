import torch
import torch.nn as nn

class invSqrt(nn.Module):
    def __init__(self, points):
        super().__init__()
        self.segments = points

    def forward(self, input):
        return input.detach().apply_(self.segments.calculate)

class NewRobertaLayerNorm(nn.Module):
      def __init__(self, weight, bias, invsqrt):
        super().__init__()
        self.weight  = weight
        self.bias    = bias
        self.invsqrt = invSqrt(invsqrt)

      def forward(self, input):
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = torch.square(input - mean).mean(dim = -1, keepdim=True)
        unaffined =  (input - mean) * self.invsqrt(var + .00001) #.00001 is epsilon, should change that
        return unaffined*self.weight + self.bias