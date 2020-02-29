# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
import os 
import sys 
from os.path import dirname 
sys.path.append("tools/")


# %%
import _init_paths
import os
import random
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor



# %%
opt_dataset_root = "./datasets/linemod/Linemod_preprocessed"
opt_model = "trained_models/linemod/pose_model_9_0.012956139583687484.pth"
opt_refine_model = "trained_models/linemod/pose_refine_model_95_0.007274364822843561.pth"

num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
iteration = 4
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'
knn = KNearestNeighbor(1)



# %%
estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt_model))
refiner.load_state_dict(torch.load(opt_refine_model))
estimator.eval()
refiner.eval()

testdataset = PoseDataset_linemod('eval', num_points, False, opt_dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0) #Change me back

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(diameter)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx, img = data
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                     Variable(choose).cuda(), \
                                                     Variable(img).cuda(), \
                                                     Variable(target).cuda(), \
                                                     Variable(model_points).cuda(), \
                                                     Variable(idx).cuda()


# %%
from  visualization import Visualizer
import matplotlib.pyplot as plt
test1 = Visualizer(testdataset)

fig, ax = plt.subplots(1,2)
ax[0].imshow(test1.RGB(0))
ax[1].imshow(test1.True_Mask(0))



# %%
pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
pred_c = pred_c.view(bs, num_points)
how_max, which_max = torch.max(pred_c, 1)
pred_t = pred_t.view(bs * num_points, 1, 3)

my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
my_pred = np.append(my_r, my_t)

for ite in range(0, iteration):
    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
    my_mat = quaternion_matrix(my_r)
    R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
    my_mat[0:3, 3] = my_t
    
    new_points = torch.bmm((points - T), R).contiguous()
    pred_r, pred_t = refiner(new_points, emb, idx)
    pred_r = pred_r.view(1, 1, -1)
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
    my_r_2 = pred_r.view(-1).cpu().data.numpy()
    my_t_2 = pred_t.view(-1).cpu().data.numpy()
    my_mat_2 = quaternion_matrix(my_r_2)
    my_mat_2[0:3, 3] = my_t_2

    my_mat_final = np.dot(my_mat, my_mat_2)
    my_r_final = copy.deepcopy(my_mat_final)
    my_r_final[0:3, 3] = 0
    my_r_final = quaternion_from_matrix(my_r_final, True)
    my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

    my_pred = np.append(my_r_final, my_t_final)
    my_r = my_r_final
    my_t = my_t_final

# Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

model_points = model_points[0].cpu().detach().numpy()
my_r = quaternion_matrix(my_r)[:3, :3]
pred = np.dot(model_points, my_r.T) + my_t


# %%
from dataset_debug import dataset
testing = dataset('eval', num_points, False, opt_dataset_root, 0.0, True)


# %%
a = (img.cpu()).squeeze()
test1.RGB(0)
#plt.imshow(a, rgb=)


# %%
obj = testdataset.list_obj[10]
rank = testdataset.list_rank[10]  


# %%
import numpy.ma as ma
ma.getmaskarray(ma.masked_equal(label, np.array(255)))
test1 = testdataset.get_bbox(mask_to_bbox(mask_label))


# %%
# try to visualize
cam_cx = 325.26110
cam_cy = 242.04899
cam_fx = 572.41140
cam_fy = 573.57043
K = np.array([[cam_fx,0,cam_cx],[0,cam_fy,cam_cy],[0,0,1]])


# %%
res = np.dot(K,pred.T)
points = res.T[:,[0,1]]
points = np.floor(points).astype(int)
firstPoint = np.squeeze(points[1])
firstPoint = tuple(firstPoint)
for pt in points: 
    print(tuple(pt))


# %%
import cv2
a = test1.RGB(0)
for pt in points:
    pt = tuple(pt)
    cv2.circle(a,pt,1,[255,0,0],1)
plt.imshow(a)


# %%
test1.RGB(0)


# %%


