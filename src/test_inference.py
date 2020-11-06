import _init_paths
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import random
from lib.model_repository import Resnet18_8s
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import estimate_voting_distribution_with_mean

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

    sym_cor_pred = sym_cor_pred.detach().cpu().numpy()[0]
    mask_pred = mask_pred.detach().cpu().numpy()[0][0]
    pts2d_map_pred = pts2d_map_pred.detach().cpu().numpy()[0]
    graph_pred = graph_pred.detach().cpu().numpy()[0]

    mask_pred[mask_pred > 0.5] = 1.
    mask_pred[mask_pred <= 0.5] = 0.

    return sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred

def draw_symmetry(image: np.ndarray, sym_cor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    ys, xs = np.nonzero(mask)
    for i_pt in random.sample([i for i in range(len(ys))], min(100, len(ys))):
        y = int(round(ys[i_pt]))
        x = int(round(xs[i_pt]))
        x_cor, y_cor = sym_cor[:, y, x]
        x_cor = int(round(x + x_cor))
        y_cor = int(round(y + y_cor))
        result = cv2.line(result, (x, y), (x_cor, y_cor), (0, 0, 255), 1)
    return result

def draw_mask(image: np.ndarray, mask: np.ndarray, color: list=[0, 0, 255]) -> np.ndarray:
    # result = image.copy()
    result = np.zeros_like(image)
    result[mask_pred != 0] = np.array(color, dtype=np.uint8)
    return result

# def vote_keypoints(pts2d_map, mask):
#     mask = (mask > 0.5).long() # convert to binary and int64 to comply with pvnet interface
#     pts2d_map = pts2d_map.permute((1, 2, 0))
#     h, w, num_keypts_2 = pts2d_map.shape
#     pts2d_map = pts2d_map.view((h, w, num_keypts_2 // 2, 2))
#     mean = ransac_voting_layer_v3(mask, pts2d_map, 512, inlier_thresh=0.99)
#     mean, var = estimate_voting_distribution_with_mean(mask, pts2d_map, mean)
#     return mean, var

# def draw_keypoints(image: np.ndarray, pts2d_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     result = image.copy()

#     # vote keypoints
#     pts2d, _ = vote_keypoints(pts2d_map, mask)
#     pts2d = pts2d.detach().cpu().numpy()[0]
#     # draw predication
#     for i in range(pts2d.shape[0]):
#         x, y = pts2d[i]
#         x = int(round(x))
#         y = int(round(y))
#         # radius=2, color=red, thickness=filled
#         result = cv2.circle(result, (x, y), 2, (0, 0, 255), thickness=-1)

#     return result

args = parse_args()
img = cv2.imread('/home/clayton/workspace/git/HybridPose/data/linemod/original_dataset/cat/data/color0.jpg')
model = get_model(args)
model.eval()
sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred = infer_image(args=args, model=model, img=img)

print(f'sym_cor_pred.shape: {sym_cor_pred.shape}')
print(f'mask_pred.shape: {mask_pred.shape}')
print(f'pts2d_map_pred.shape: {pts2d_map_pred.shape}')
print(f'graph_pred.shape: {graph_pred.shape}')

symmetry_preview = draw_symmetry(image=img, sym_cor=sym_cor_pred, mask=mask_pred)
cv2.imwrite('symmetry_preview.png', symmetry_preview)

mask_preview = draw_mask(image=img, mask=mask_pred, color=[0, 0, 255])
cv2.imwrite('mask_preview.png', mask_preview)

# pts2d_map_tensor = torch.from_numpy(pts2d_map_pred).reshape(tuple([1]+list(pts2d_map_pred.shape)))
# mask_tensor = torch.from_numpy(mask_pred)
# keypoints_preview = draw_keypoints(image=img, pts2d_map=pts2d_map_tensor, mask=mask_tensor)
# cv2.imwrite('keypoints_preview.png', keypoints_preview)