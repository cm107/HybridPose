import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import KDTree
from HybridPose.lib.datasets.custom_blender_linemod import BlenderLinemodDataset
from HybridPose.lib.regressor.regressor import load_wrapper
from HybridPose.lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from HybridPose.lib.ransac_voting_gpu_layer.ransac_voting_gpu import estimate_voting_distribution_with_mean
from HybridPose.src.evaluate import custom_read_diameter
from HybridPose.custom_src.utils import get_model
from HybridPose.lib.regressor.regressor import load_wrapper, get_2d_ctypes

def vote_keypoints(pts2d_map, mask):
    mask = mask[:, 0] # remove dummy dimension
    mask = (mask > 0.5).long() # convert to binary and int64 to comply with pvnet interface
    pts2d_map = pts2d_map.permute((0, 2, 3, 1))
    bs, h, w, num_keypts_2 = pts2d_map.shape
    pts2d_map = pts2d_map.view((bs, h, w, num_keypts_2 // 2, 2))
    mean = ransac_voting_layer_v3(mask, pts2d_map, 512, inlier_thresh=0.99)
    mean, var = estimate_voting_distribution_with_mean(mask, pts2d_map, mean)
    return mean, var

def flatten_sym_cor(sym_cor, mask):
    ys, xs = np.nonzero(mask)
    flat = np.zeros((ys.shape[0], 2, 2), dtype=np.float32)
    for i_pt in range(len(ys)):
        y = ys[i_pt]
        x = xs[i_pt]
        x_cor, y_cor = sym_cor[:, y, x]
        flat[i_pt, 0] = [x, y]
        flat[i_pt, 1] = [x + x_cor, y + y_cor]
    return flat

def filter_symmetry(vecs_pred, sigma=0.01, min_count=100, n_neighbors=100):
    # Chen: I have to set min_count >= neighbors here.
    #       Otherwise kdtree will complain "k must be less than or equal to the number of training points"
    if len(vecs_pred) < min_count:
        qs1_cross_qs2 = np.zeros((0, 3), dtype=np.float32)
        symmetry_weight = np.zeros((0,), dtype=np.float32)
        return qs1_cross_qs2, symmetry_weight
    vecs_pred /= np.sqrt(np.sum(vecs_pred[:, :2]**2, axis=1)).reshape((-1, 1))
    kdt = KDTree(vecs_pred, leaf_size=40, metric='euclidean') # following matlab default values
    dis, _ = kdt.query(vecs_pred, k=n_neighbors)
    saliency = np.mean(dis * dis, axis=1, dtype=np.float32)
    order = np.argsort(saliency)
    seeds = np.zeros((2, order.shape[0]), dtype=np.uint32)
    seeds[0][0] = order[0]
    seeds[1][0] = 1
    seeds_size = 1
    flags = np.zeros((order.shape[0],), dtype=np.uint32)
    flags[order[0]] = 0
    for i in range(1, order.shape[0]):
        vec = vecs_pred[order[i]]
        candidates = vecs_pred[seeds[0]]
        dif = candidates - vec
        norm = np.linalg.norm(dif, axis=1)
        closest_seed_i = norm.argmin()
        min_dis = norm[closest_seed_i]
        if min_dis < sigma:
            flags[order[i]] = closest_seed_i
            seeds[1][closest_seed_i] = seeds[1][closest_seed_i] + 1
        else:
            seeds[0, seeds_size] = order[i]
            seeds[1, seeds_size] = 1
            flags[order[i]] = seeds_size
            seeds_size += 1
    seeds = seeds[:, :seeds_size]
    valid_is = np.argwhere(seeds[1] > (np.max(seeds[1]) / 3)).transpose()[0]
    seeds = seeds[:, valid_is]
    n_symmetry = seeds.shape[1]
    qs1_cross_qs2 = np.zeros((n_symmetry, 3), dtype=np.float32)
    for i in range(n_symmetry):            
        row_is = np.argwhere(flags == valid_is[i]).transpose()[0]
        qs1_cross_qs2[i] = np.mean(vecs_pred[row_is], axis=0)
        qs1_cross_qs2[i] /= np.linalg.norm(qs1_cross_qs2[i])
    symmetry_weight = np.float32(seeds[1])
    symmetry_weight /= np.max(symmetry_weight)
    return qs1_cross_qs2, symmetry_weight

def fill_intermediate_predictions(regressor, predictions, K_inv, pts3d, pts2d_pred_loc, pts2d_pred_var, graph_pred, sym_cor_pred, mask_pred, normal_gt, num_keypoints: int):
    # load intermediate representations to regressor
    n_keypts = num_keypoints
    n_edges = n_keypts * (n_keypts - 1) // 2
    # point3D_gt
    regressor.set_point3D_gt(predictions, get_2d_ctypes(pts3d), n_keypts)
    # point2D_pred
    point2D_pred = np.matrix(np.ones((3, n_keypts), dtype=np.float32))
    point2D_pred[:2] = pts2d_pred_loc.transpose()
    point2D_pred = np.array((K_inv * point2D_pred)[:2]).transpose()
    regressor.set_point2D_pred(predictions,
                                get_2d_ctypes(point2D_pred),
                                n_keypts)
    # point_inv_half_var
    point_inv_half_var = np.zeros((n_keypts, 2, 2), dtype=np.float32)
    for i in range(n_keypts): # compute cov^{-1/2}
        cov = np.matrix(pts2d_pred_var[i])
        cov = (cov + cov.transpose()) / 2 # ensure the covariance matrix is symmetric
        v, u = np.linalg.eig(cov)
        v = np.matrix(np.diag(1. / np.sqrt(v)))
        point_inv_half_var[i] = u * v * u.transpose()
    point_inv_half_var = point_inv_half_var.reshape((n_keypts, 4))
    regressor.set_point_inv_half_var(predictions,
                                        get_2d_ctypes(point_inv_half_var),
                                        n_keypts)
    # normal_gt
    regressor.set_normal_gt(predictions, normal_gt.ctypes)
    # vec_pred and edge_inv_half_var
    graph_pred = graph_pred.reshape((n_edges, 2, graph_pred.shape[1], graph_pred.shape[2]))
    vec_pred = np.zeros((n_edges, 2), dtype=np.float32)
    edge_inv_half_var = np.zeros((n_edges, 2, 2), dtype=np.float32)
    for i in range(n_edges):
        xs = graph_pred[i, 0][mask_pred == 1.]
        ys = graph_pred[i, 1][mask_pred == 1.]
        vec_pred[i] = [xs.mean(), ys.mean()]
        try:
            cov = np.cov(xs, ys)
            cov = (cov + cov.transpose()) / 2 # ensure the covariance matrix is symmetric
            v, u = np.linalg.eig(cov)
            v = np.matrix(np.diag(1. / np.sqrt(v)))
            edge_inv_half_var[i] = u * v * u.transpose()
        except:
            edge_inv_half_var[i] = np.eye(2)
    vec_pred = np.array(K_inv[:2, :2] * np.matrix(vec_pred).transpose()).transpose()
    edge_inv_half_var = edge_inv_half_var.reshape((n_edges, 4))
    regressor.set_vec_pred(predictions,
                            get_2d_ctypes(vec_pred),
                            n_edges)
    regressor.set_edge_inv_half_var(predictions,
                                    get_2d_ctypes(edge_inv_half_var),
                                    n_edges)
    # qs1_cross_qs2 and symmetry weight
    sym_cor_pred = flatten_sym_cor(sym_cor_pred, mask_pred)

    qs1_cross_qs2_all = np.zeros((sym_cor_pred.shape[0], 3), dtype=np.float32)
    for i in range(sym_cor_pred.shape[0]):
        qs1 = np.ones((3,), dtype=np.float32)
        qs2 = np.ones((3,), dtype=np.float32)
        qs1[:2] = sym_cor_pred[i][0]
        qs2[:2] = sym_cor_pred[i][1]
        qs1 = np.array(K_inv * np.matrix(qs1).transpose()).transpose()[0]
        qs2 = np.array(K_inv * np.matrix(qs2).transpose()).transpose()[0]
        qs1_cross_qs2_all[i] = np.cross(qs1, qs2)
    qs1_cross_qs2_filtered, symmetry_weight = filter_symmetry(qs1_cross_qs2_all)
    n_symmetry = qs1_cross_qs2_filtered.shape[0]
    regressor.set_qs1_cross_qs2(predictions,
                                get_2d_ctypes(qs1_cross_qs2_filtered),
                                n_symmetry)
    regressor.set_symmetry_weight(predictions,
                                    symmetry_weight.ctypes,
                                    n_symmetry)  

def regress_pose(regressor, predictions, pr_para, pi_para, K_inv, pts3d, pts2d_pred_loc, pts2d_pred_var, graph_pred, sym_cor_pred, mask_pred, normal_gt, num_keypoints: int):
    if mask_pred.sum() == 0:
        # object is not detected
        R = np.eye(3, dtype=np.float32)
        t = np.zeros((3, 1), dtype=np.float32)
        return R, t, R, t
    fill_intermediate_predictions(regressor,
                                    predictions,
                                    K_inv,
                                    pts3d,
                                    pts2d_pred_loc,
                                    pts2d_pred_var,
                                    graph_pred,
                                    sym_cor_pred,
                                    mask_pred,
                                    normal_gt,
                                    num_keypoints)
    # initialize pose
    predictions = regressor.initialize_pose(predictions, pi_para)
    pose_init = np.zeros((4, 3), dtype=np.float32)
    regressor.get_pose(predictions, get_2d_ctypes(pose_init))
    R_init = pose_init[1:].transpose()
    t_init = pose_init[0].reshape((3, 1))
    # refine pose
    predictions = regressor.refine_pose(predictions, pr_para)
    pose_final = np.zeros((4, 3), dtype=np.float32)
    regressor.get_pose(predictions, get_2d_ctypes(pose_final))
    R_final = pose_final[1:].transpose()
    t_final = pose_final[0].reshape((3, 1))
    return R_final, t_final, R_init, t_init

def search_para(regressor, predictions_para, poses_para, K_inv, normal_gt, diameter, val_set, num_keypoints):
    para_id = 0
    for data_id in range(len(val_set['pts3d'])):
        if val_set['mask_pred'][data_id].sum() == 0 or \
                np.sum(val_set['pts2d_pred_loc'][data_id]) == 0:
            # object not detected
            continue
        predictions = regressor.get_prediction_container(predictions_para, para_id)
        # fill intermediate predictions
        fill_intermediate_predictions(regressor,
                                        predictions,
                                        K_inv,
                                        val_set['pts3d'][data_id],
                                        val_set['pts2d_pred_loc'][data_id],
                                        val_set['pts2d_pred_var'][data_id],
                                        val_set['graph_pred'][data_id],
                                        val_set['sym_cor_pred'][data_id],
                                        val_set['mask_pred'][data_id],
                                        normal_gt,
                                        num_keypoints)
        # fill ground-truth poses
        pose_gt = np.zeros((4, 3), dtype=np.float32)
        tvec = val_set['t_gt'][data_id]
        r = val_set['R_gt'][data_id]
        pose_gt[0] = tvec.transpose()[0]
        pose_gt[1:] = r.transpose()
        regressor.set_pose_gt(poses_para, para_id, get_2d_ctypes(pose_gt))
        # increment number of valid examples in the val set
        para_id += 1
    # search parameter
    # para_id is datasize for parameter search
    pi_para = regressor.search_pose_initial(predictions_para, poses_para, para_id, diameter)
    pr_para = regressor.search_pose_refine(predictions_para, poses_para, para_id, diameter)
    return pr_para, pi_para

def search_for_params(regressor, val_set, val_loader: DataLoader, batch_size: int, K_inv, predictions_para, poses_para, val_size: int, diameter_path: str, num_keypoints: int=8, use_cuda: bool=True):
    # search parameters
    keep_searching = True
    for i_batch, batch in enumerate(val_loader):
        if not keep_searching:
            break
        base_idx = batch_size * i_batch
        if use_cuda:
            batch['image'] = batch['image'].cuda()
            batch['sym_cor'] = batch['sym_cor'].cuda()
            batch['mask'] = batch['mask'].cuda()
            batch['pts2d_map'] = batch['pts2d_map'].cuda()
            batch['graph'] = batch['graph'].cuda()
        sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred, sym_cor_loss, mask_loss, pts2d_loss, graph_loss = \
                model(batch['image'], batch['sym_cor'], batch['mask'], batch['pts2d_map'], batch['graph'])
        mask_pred[mask_pred > 0.5] = 1.
        mask_pred[mask_pred <= 0.5] = 0.
        pts2d_pred_loc, pts2d_pred_var = vote_keypoints(pts2d_map_pred, mask_pred)
        mask_pred = mask_pred.detach().cpu().numpy()     
        for i in range(batch['image'].shape[0]):
            R = batch['R'].numpy()
            t = batch['t'].numpy()

            if (base_idx + i) < val_size:
                # save data for parameter search                    
                val_set['pts3d'].append(batch['pts3d'][i].numpy())
                val_set['pts2d_pred_loc'].append(pts2d_pred_loc[i].detach().cpu().numpy())
                val_set['pts2d_pred_var'].append(pts2d_pred_var[i].detach().cpu().numpy())
                val_set['graph_pred'].append(graph_pred[i].detach().cpu().numpy())
                val_set['sym_cor_pred'].append(sym_cor_pred[i].detach().cpu().numpy())
                val_set['mask_pred'].append(mask_pred[i][0]) 
                val_set['R_gt'].append(R[i])
                val_set['t_gt'].append(t[i])
            elif (base_idx + i) == val_size:
                # search hyper-parameters of both initialization and refinement sub-modules
                pr_para, pi_para = search_para(regressor,
                                                    predictions_para,
                                                    poses_para,
                                                    K_inv,
                                                    batch['normal'][i].numpy(),
                                                    custom_read_diameter(diameter_path),
                                                    val_set, num_keypoints)
                keep_searching = False
                break
    return pr_para, pi_para

def generate_data(
    model: nn.Module, val_loader: DataLoader, test_loader: DataLoader, diameter_path: str,
    val_size=50, dump_path: str='dump.npy', num_keypoints: int=8
):
    print(f'Flag0')
    assert test_loader.batch_size == val_loader.batch_size
    use_cuda = torch.cuda.is_available()
    batch_size = test_loader.batch_size
    model.eval()
    camera_intrinsic = test_loader.dataset.camera_intrinsic
    n_examples = len(test_loader.dataset)
    test_set = {
            'object_name': [],
            'local_idx': [],
            'R_gt': np.zeros((n_examples, 3, 3), dtype=np.float32),
            't_gt': np.zeros((n_examples, 3, 1), dtype=np.float32),
            'R_pred': np.zeros((n_examples, 3, 3), dtype=np.float32),
            't_pred': np.zeros((n_examples, 3, 1), dtype=np.float32),
            'R_init': np.zeros((n_examples, 3, 3), dtype=np.float32),
            't_init': np.zeros((n_examples, 3, 1), dtype=np.float32)
            }  
    val_set = {
                'pts3d' : [],
                'pts2d_pred_loc' : [],
                'pts2d_pred_var' : [],
                'graph_pred' : [],
                'sym_cor_pred' : [],
                'mask_pred' : [],
                'R_gt' : [],
                't_gt' : []
                }
    K = np.matrix([[camera_intrinsic['fu'], 0, camera_intrinsic['uc']],
                    [0, camera_intrinsic['fv'], camera_intrinsic['vc']],
                    [0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)
    regressor = load_wrapper()
    # intermediate predictions in the test set
    predictions = regressor.new_container()
    # intermediate predictions in the val set
    predictions_para = regressor.new_container_para()
    # ground-truth poses in the val set
    poses_para = regressor.new_container_pose()
    with torch.no_grad():
        # # search parameters
        # keep_searching = True
        # for i_batch, batch in enumerate(val_loader):
        #     if not keep_searching:
        #         break
        #     base_idx = batch_size * i_batch
        #     if use_cuda:
        #         batch['image'] = batch['image'].cuda()
        #         batch['sym_cor'] = batch['sym_cor'].cuda()
        #         batch['mask'] = batch['mask'].cuda()
        #         batch['pts2d_map'] = batch['pts2d_map'].cuda()
        #         batch['graph'] = batch['graph'].cuda()
        #     sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred, sym_cor_loss, mask_loss, pts2d_loss, graph_loss = \
        #             model(batch['image'], batch['sym_cor'], batch['mask'], batch['pts2d_map'], batch['graph'])
        #     mask_pred[mask_pred > 0.5] = 1.
        #     mask_pred[mask_pred <= 0.5] = 0.
        #     pts2d_pred_loc, pts2d_pred_var = vote_keypoints(pts2d_map_pred, mask_pred)
        #     mask_pred = mask_pred.detach().cpu().numpy()     
        #     for i in range(batch['image'].shape[0]):
        #         R = batch['R'].numpy()
        #         t = batch['t'].numpy()

        #         if (base_idx + i) < val_size:
        #             # save data for parameter search                    
        #             val_set['pts3d'].append(batch['pts3d'][i].numpy())
        #             val_set['pts2d_pred_loc'].append(pts2d_pred_loc[i].detach().cpu().numpy())
        #             val_set['pts2d_pred_var'].append(pts2d_pred_var[i].detach().cpu().numpy())
        #             val_set['graph_pred'].append(graph_pred[i].detach().cpu().numpy())
        #             val_set['sym_cor_pred'].append(sym_cor_pred[i].detach().cpu().numpy())
        #             val_set['mask_pred'].append(mask_pred[i][0]) 
        #             val_set['R_gt'].append(R[i])
        #             val_set['t_gt'].append(t[i])
        #         elif (base_idx + i) == val_size:
        #             # search hyper-parameters of both initialization and refinement sub-modules
        #             pr_para, pi_para = search_para(regressor,
        #                                                 predictions_para,
        #                                                 poses_para,
        #                                                 K_inv,
        #                                                 batch['normal'][i].numpy(),
        #                                                 custom_read_diameter(diameter_path),
        #                                                 val_set, num_keypoints)
        #             keep_searching = False
        #             break
        pr_para, pi_para = search_for_params(regressor, val_set, val_loader, batch_size, K_inv, predictions_para, poses_para, val_size, diameter_path, num_keypoints, use_cuda)
        # prediction
        for i_batch, batch in enumerate(test_loader):
            base_idx = batch_size * i_batch
            if use_cuda:
                batch['image'] = batch['image'].cuda()
                batch['sym_cor'] = batch['sym_cor'].cuda()
                batch['mask'] = batch['mask'].cuda()
                batch['pts2d_map'] = batch['pts2d_map'].cuda()
                batch['graph'] = batch['graph'].cuda()
            sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred, sym_cor_loss, mask_loss, pts2d_loss, graph_loss = \
                    model(batch['image'], batch['sym_cor'], batch['mask'], batch['pts2d_map'], batch['graph'])
            mask_pred[mask_pred > 0.5] = 1.
            mask_pred[mask_pred <= 0.5] = 0.
            pts2d_pred_loc, pts2d_pred_var = vote_keypoints(pts2d_map_pred, mask_pred)
            mask_pred = mask_pred.detach().cpu().numpy()
            for i in range(batch['image'].shape[0]):
                R = batch['R'].numpy()
                t = batch['t'].numpy()
                # regress pose: test set starts from the `val_size`^{th} example
                # save ground-truth information
                test_set['object_name'] += batch['object_name'][i:]
                test_set['local_idx'] += batch['local_idx'].numpy()[i:].tolist()
                test_set['R_gt'][base_idx + i] = R[i]
                test_set['t_gt'][base_idx + i] = t[i] if t[i].shape != (3,) else t[i].reshape(-1, 1)
                # save predicted information
                R_pred, t_pred, R_init, t_init = regress_pose(regressor,
                                                                predictions,
                                                                pr_para,
                                                                pi_para,
                                                                K_inv,
                                                                batch['pts3d'][i].numpy(),
                                                                pts2d_pred_loc[i].detach().cpu().numpy(),
                                                                pts2d_pred_var[i].detach().cpu().numpy(),
                                                                graph_pred[i].detach().cpu().numpy(),
                                                                sym_cor_pred[i].detach().cpu().numpy(),
                                                                mask_pred[i][0],
                                                                batch['normal'][i].numpy(),
                                                                num_keypoints)
                test_set['R_pred'][base_idx + i] = R_pred
                test_set['t_pred'][base_idx + i] = t_pred
                test_set['R_init'][base_idx + i] = R_init
                test_set['t_init'][base_idx + i] = t_init
        np.save(dump_path, test_set)
        print('saved')
    regressor.delete_container(predictions, predictions_para, poses_para, pr_para, pi_para)

val_set = BlenderLinemodDataset(
    data_dir='custom_renders_val/hsr',
    labels_dir='custom_renders_val/hsr_labels',
    keypoints_3d_path='hsr_keypoints_3d.npy',
    symmetries_path='hsr_symmetries.txt',
    object_name='hsr',
    size=200,
    augment=False,
    occlude=False,
    split='val'
)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
test_set = BlenderLinemodDataset(
    data_dir='custom_renders_test/hsr',
    labels_dir='custom_renders_test/hsr_labels',
    keypoints_3d_path='hsr_keypoints_3d.npy',
    symmetries_path='hsr_symmetries.txt',
    object_name='hsr',
    size=200,
    augment=False,
    occlude=False,
    split='test'
)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
model = get_model(weights_path='weights/model.pth', num_keypoints=8, use_cuda=True)
generate_data(
    model=model, val_loader=val_loader, test_loader=test_loader,
    diameter_path='diameter.txt', val_size=50,
    dump_path='dump.npy',
    num_keypoints=8
)