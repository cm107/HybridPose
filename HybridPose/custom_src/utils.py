import torch
import torch.nn as nn
from ..lib.model_repository import Resnet18_8s

def load_model(model: nn.Module, weights_path: str) -> nn.Module:
    model.load_state_dict(torch.load(weights_path))
    return model

def get_model(weights_path: str, num_keypoints: int=8, use_cuda: bool=True) -> nn.Module:
    model = Resnet18_8s(num_keypoints=num_keypoints)
    if use_cuda:
        model = nn.DataParallel(model).cuda()
    model = load_model(model=model, weights_path=weights_path)
    return model