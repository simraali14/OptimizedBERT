
import os
import sys
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)
from utils.weight_bit_precision_editor import WeightBitPrecisionEditor
from utils.piecewise_activation import PieceWiseActivation

def apply_half_precision(model):
    with torch.no_grad():
        model = model.half()
    return model

def apply_piecewise_approximation(model):
    with torch.no_grad():
        for layer in model.roberta.encoder.layer:
            layer.intermediate.intermediate_act_fn = PieceWiseActivation()
    return model

def apply_edit_bit_precisionclear(model, bits):
    from utils.weight_bit_precision_editor import WeightBitPrecisionEditor
    weight_editor = WeightBitPrecisionEditor(model, bits)
    with torch.no_grad():
        weight_editor.shave_weights()
        model = weight_editor.get_model()
    return model