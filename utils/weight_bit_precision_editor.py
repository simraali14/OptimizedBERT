import numpy
import torch 

class WeightBitPrecisionEditor:
    """
    A class to edit the weights of a PyTorch model by converting them 
    to a fixed-point representation with a specified number of bits.
    
    Attributes:
        model (torch.nn.Module): The PyTorch model whose weights will be edited.
        bits (int): The number of bits to use for the fixed-point representation.
        all_keys (list): A list of all keys in the model's state dictionary.
        encoder_keys (list): A subset of keys representing weights related 
            to the encoder part of the model.
    """
    def __init__(self, model, bits):
        self.model = model
        self.bits = bits
        self.all_keys = list(model.state_dict().keys())
        self.encoder_keys = [x for x in self.all_keys if "attention.self" in x and 'insert_other_cond' not in x]

    def get_model(self):
        return self.model

    def to_float(self,x,e):
        c = abs(x)
        sign = 1
        if x < 0:
            c = x - 1
            c = ~c
            sign = -1
        f = (1.0 * c) / (2 ** e)
        f = f * sign
        return f

    def to_fixed(self, f,e):
        a = f* (2**e)
        b = int(round(a))
        if a < 0:
            b = abs(b)
            b = ~b
            b = b + 1
        return b

    def shave_weights(self):
        with torch.no_grad():
            for k in self.encoder_keys:
                if 'weight' in k:
                    for x in self.model.state_dict()[k]:
                        for value in range(0,len(x)):
                            weight = x[value].item()
                            x[value] = torch.tensor(self.to_float(self.to_fixed(weight,self.bits),self.bits))
                else: #bias
                    for x in range(0,len(self.model.state_dict()[k])):
                        weight = self.model.state_dict()[k][x].item()
                        self.model.state_dict()[k][x] = torch.tensor(self.to_float(self.to_fixed(weight,self.bits),self.bits))