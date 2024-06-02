
import warnings
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import torch
import logging
import pickle
from transformers import AutoModelForMaskedLM
import optimize_model

def save_model(model, filename):
    os.makedirs('./models', exist_ok=True)
    with open(f'./models/{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}.pkl")

# Setup
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
model_checkpoint = "distilroberta-base"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# Optimize with approximation and editing functions
#model = optimize_model.apply_half_precision(model)
model = optimize_model.apply_piecewise_approximation(model)
model = optimize_model.apply_edit_bit_precisionclear(model, 8)

# Saving model
save_model(model, "piecewise_with_eight_bits_model")