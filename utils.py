import torch
import random
import numpy as np
import os

def set_seed(seed):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_model(model, model_path, epoch):
    os.makedirs(model_path, exist_ok=True)
    state = {
        'model': model.state_dict()
    }
    torch.save(state, os.path.join(model_path, f'model{epoch}.pth'))


def load_model(model, model_path, epoch):
    state = torch.load(os.path.join(model_path, f'model{epoch}.pth'))
    model.load_state_dict(state['model'])