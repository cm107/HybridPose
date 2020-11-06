import _init_paths
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from lib.model_repository import Resnet18_8s

def load_model(model, args):
    model.load_state_dict(torch.load(args.weights_path))
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--weights_path', type=str)
    parser.add_argument('--num_keypoints', type=int, default=8)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()
    return args

def get_model(args):
    model = Resnet18_8s(num_keypoints=args.num_keypoints)
    if args.cuda:
        model = nn.DataParallel(model).cuda()
    model = load_model(model, args)
    return model

def infer_image(args, model: nn.Module, img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    image = img.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
    image = torch.from_numpy(image)
    image = image.reshape(tuple([1] + list(image.shape)))
    image = image.float()
    if args.normalize:
        from torchvision import transforms
        img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
        image = img_transform(image)
    image = image.cuda()

    sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred = model.module.predict(image=image)

    sym_cor_pred = sym_cor_pred.detach().cpu().numpy()
    mask_pred = mask_pred.detach().cpu().numpy()
    pts2d_map_pred = pts2d_map_pred.detach().cpu().numpy()
    graph_pred = graph_pred.detach().cpu().numpy()

    return sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred

args = parse_args()
img = cv2.imread('/home/clayton/workspace/git/HybridPose/data/linemod/original_dataset/cat/data/color0.jpg')
model = get_model(args)
model.eval()
sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred = infer_image(args=args, model=model, img=img)

print(f'sym_cor_pred: {sym_cor_pred}')
print(f'mask_pred: {mask_pred}')
print(f'pts2d_map_pred: {pts2d_map_pred}')
print(f'graph_pred: {graph_pred}')
