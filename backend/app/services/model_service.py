import torch
from app.models.simple_nn import SimpleNN

# Initialize the model
model = SimpleNN()

def get_model_params():
    """
    Extracts the weights and biases from the model and returns them as a dictionary.
    """
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.detach().numpy().tolist()
    return params
