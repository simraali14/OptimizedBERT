
import os
import sys
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)
from utils.piecewise_activation import PieceWiseActivation
from utils.weight_bit_precision_editor import WeightBitPrecisionEditor
from utils.layernorm import NewRobertaLayerNorm
from utils.softmax import NewRobertaSelfAttention

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
    weight_editor = WeightBitPrecisionEditor(model, bits)
    with torch.no_grad():
        weight_editor.shave_weights()
        model = weight_editor.get_model()
    return model

def apply_layernorm(model):
    for i in range(0, len(model.roberta.encoder.layer)):
        w,b = model.roberta.encoder.layer[i].output.LayerNorm.weight, model.roberta.encoder.layer[i].output.LayerNorm.bias
        model.roberta.encoder.layer[i].output.LayerNorm = NewRobertaLayerNorm(w,b, invsqrt)

def apply_softmax(model):
    for x in range(0, len(model.roberta.encoder.layer)):
        model.roberta.encoder.layer[x].attention.self = NewRobertaSelfAttention(model.config)
        model.roberta.encoder.layer[x].attention.self.load_state_dict(model.roberta.encoder.layer[x].attention.self.state_dict())