
import os
import sys
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)
from utils.piecewise import Piecewise
from utils.math_functions import exponential, reciprocal, reciprocal_prime, inverse_sqrt, inverse_sqrt_prime, gelu, gelu_prime

from utils.piecewise_activation import PiecewiseActivation
from utils.weight_bit_precision_editor import WeightBitPrecisionEditor
from utils.layernorm import PiecewiseLayerNorm
from utils.softmax import NewRobertaSelfAttention

# approximation specifications
piecewise_approximations = {"gelu" : Piecewise(-10, 10, gelu, gelu_prime),
                            "inverse_sqrt" : Piecewise(0.00001, 150, inverse_sqrt, inverse_sqrt_prime),
                             "exponential" :  Piecewise(-100, 100, exponential, exponential),
                             "reciprocal" : Piecewise(0.01, 764, reciprocal, reciprocal_prime)}

def apply_half_precision(model):
    with torch.no_grad():
        model = model.half()
    return model

def approximate_gelu_activation(model):
    with torch.no_grad():
        for layer in model.roberta.encoder.layer:
            layer.intermediate.intermediate_act_fn = PiecewiseActivation(piecewise_approximations["gelu"])
    return model

def approximate_layernorm(model):
    for i in range(0, len(model.roberta.encoder.layer)):
        w,b = model.roberta.encoder.layer[i].output.LayerNorm.weight, model.roberta.encoder.layer[i].output.LayerNorm.bias
        model.roberta.encoder.layer[i].output.LayerNorm = PiecewiseLayerNorm(w,b,piecewise_approximations["inverse_sqrt"])
    return model

def approximate_softmax(model):
    for x in range(0, len(model.roberta.encoder.layer)):
        model.roberta.encoder.layer[x].attention.self = NewRobertaSelfAttention(model.config,piecewise_approximations["exponential"],piecewise_approximations["reciprocal"])
        model.roberta.encoder.layer[x].attention.self.load_state_dict(model.roberta.encoder.layer[x].attention.self.state_dict())
    return model

def edit_bit_precision(model, bits):
    weight_editor = WeightBitPrecisionEditor(model, bits)
    with torch.no_grad():
        weight_editor.shave_weights()
        model = weight_editor.get_model()
    return model

def save_piecewise_approximations():
    for name,approx in piecewise_approximations.items():
        file_path = f"{name}_piecewise_approximation.txt"
        with open(file_path, 'w') as txt_file:
            for key, value in approx.segment_data.items():
                txt_file.write(f"{key}: {value}\n")
        print(f"Piecewise segments for {name} saved to {file_path}")