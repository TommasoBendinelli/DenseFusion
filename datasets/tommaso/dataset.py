import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
from scipy.interpolate import griddata
import cv2
import trimesh
import open3d as o3d
import data_augmentation
from skimage import transform as tf

class PoseDataset(data.Dataset):
    def __init__(self, mode, num,root, refine,add_noise=False ,noise_trans = None, is_visualized = False):
        self.objlist = [1]
        self.mode = mode
        self.issue_counter = 0
        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        #self.pt = {}
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine
        self.mesh = {}
        self.is_visualized = is_visualized
        t = np.loadtxt('datasets/tommaso/tommaso_preprocessed/data/01/train.txt')
        np.random.shuffle(t)
        np.savetxt('datasets/tommaso/tommaso_preprocessed/data/01/train_shuffle.txt',t.astype(int),fmt='%i')
        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train_shuffle.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}_testing/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test':
                    if not input_line:
                        break
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    self.list_rgb.append('{0}/data/{1}_testing/rgb/{2}.jpg'.format(self.root, '%02d' % item, input_line))
                    self.list_depth.append('{0}/data/{1}_testing/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                    self.list_label.append('{0}/data/{1}_testing/mask/{2}.png'.format(self.root, '%02d' % item, input_line))
                    meta_file = open('{0}/data/{1}_testing/gt.json'.format(self.root, '%02d' % item), 'r')
                else:                    
                    # if self.mode == 'test' and item_count % 10 != 0:
                    #     continue
                    if not input_line:
                        break
                    # if input_line[0] != '4':
                    #     continue
                    
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    self.list_rgb.append('{0}/data/{1}/rgb/{2}.jpg'.format(self.root, '%02d' % item, input_line))
                    self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                    #FIX ME
                    if self.mode == 'fixme':
                        self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                    else: 
                        self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))

                    meta_file = open('{0}/data/{1}/gt.json'.format(self.root, '%02d' % item), 'r')
                    
                self.list_obj.append(item)
                self.list_rank.append(int(input_line))

            
            self.meta[item] = json.load(meta_file)
            # self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            # self.mesh[item] = trimesh.load('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            self.mesh[item] = o3d.io.read_triangle_mesh('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            # self.mesh[item].apply_obb()
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.cam_cx = 323.3623962402344
        self.cam_cy = 247.32833862304688
        self.cam_fx = 614.28125
        self.cam_fy = 614.4807739257812
        self.height = 480
        self.width = 640

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.noise_params = {
    
    'max_augmentation_tries' : 10,
    
    # Padding
    'padding_alpha' : 1.0,
    'padding_beta' : 4.0, 
    'min_padding_percentage' : 0.05, 
    
    # Erosion/Dilation
    'rate_of_morphological_transform' : 0.9,
    'label_dilation_alpha' : 1.0,
    'label_dilation_beta' : 19.0,
    'morphology_max_iters' : 3,
    
    # Ellipses
    'rate_of_ellipses' : 0.8,
    'num_ellipses_mean' : 50,
    'ellipse_gamma_base_shape' : 1.0, 
    'ellipse_gamma_base_scale' : 1.0,
    'ellipse_size_percentage' : 0.025,
    
    # Translation
    'rate_of_translation' : 0.7,
    'translation_alpha' : 1.0,
    'translation_beta' : 19.0,
    'translation_percentage_min' : 0.02,
    
    # Rotation
    'rate_of_rotation' : 0.7,
    'rotation_angle_max' : 30, # in degrees
    
    # Label Cutting
    'rate_of_label_cutting' : 0.1,
    'cut_percentage_min' : 0.05,
    'cut_percentage_max' : 0.4,
    
    # Label Adding
    'rate_of_label_adding' : 0.5,
    'add_percentage_min' : 0.1,
    'add_percentage_max' : 0.4,

    # Multiplicative noise
    'gamma_shape' : 1000.,
    'gamma_scale' : 0.001,
    
    # Additive noise
    'gaussian_scale' : 0.005, # 5mm standard dev
    'gp_rescale_factor' : 4,
    
    # Random ellipse dropout
    'ellipse_dropout_mean' : 10, 
    'ellipse_gamma_shape' : 5.0, 
    'ellipse_gamma_scale' : 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean' : 15, 
    'gradient_dropout_alpha' : 2., 
    'gradient_dropout_beta' : 5.,

    # Random pixel dropout
    'pixel_dropout_alpha' : 1., 
    'pixel_dropout_beta' : 10.,
    
}
        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = num
        self.num_pt_mesh_small = num
        self.symmetry_obj_idx = [7, 8]

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])

        #img = np.fliplr(img)

        ori_img = np.array(img)
        
        #ori_img = np.fliplr(ori_img)

        depth = np.array(Image.open(self.list_depth[index]))

        #depth = np.fliplr(depth)
        # test_2 = 0
        # if not test_2:
        #     depth = np.where(depth==0, np.random.uniform(0,8), depth)
        depth = np.where(depth==0, np.random.uniform(0,8), depth)
        # skew_tran = 1
        # if perspective_transform:
            
        #     a,b,c,d = round(random.uniform(0,360)), 
        #     M = cv2.getPerspectiveTransform(pts1,pts2)
        #     dst = cv2.warpPerspective(img,M,(300,300))

        if self.add_noise:
            pass
            #depth = depth + np.random.normal(scale=5,size=[480,640])
            #depth = data_augmentation.add_noise_to_depth(depth, self.noise_params)
            #depth = data_augmentation.dropout_random_ellipses(depth, self.noise_params)

        label = np.array(Image.open(self.list_label[index]))

        mask_label_old = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label_old))
        try:
            # shear_transform = 1
            # if shear_transform:
            #     # Create Afine transform
            #     affine_tf = tf.AffineTransform(shear=0.4)
            #     ori_img = np.round(tf.warp(ori_img/255,inverse_map=affine_tf)*255).astype('uint8')
            #     img = np.round(tf.warp(np.asarray(img)/255,inverse_map=affine_tf)*255).astype('uint8')
            #     depth = np.round(tf.warp(depth/depth.max(),inverse_map=affine_tf)*depth.max()).astype('float64')
            #     label = np.round(tf.warp(label/255,inverse_map=affine_tf)*255).astype('uint8')
            if self.add_noise:
                img = self.trancolor(img)

            if self.add_noise: 
                tx_f, ty_f = [round(random.uniform(-150,150)), round(random.uniform(-150,150))]
                ori_img = data_augmentation.translate(ori_img, tx_f, ty_f, interpolation=cv2.INTER_LINEAR)
                img = data_augmentation.translate(np.asarray(img), tx_f, ty_f, interpolation=cv2.INTER_LINEAR)
                depth = data_augmentation.translate(depth, tx_f, ty_f, interpolation=cv2.INTER_LINEAR)
                label = data_augmentation.translate(label, tx_f, ty_f, interpolation=cv2.INTER_LINEAR)


            if self.add_noise:
                rotation = round(random.uniform(0,360))
                ori_img = data_augmentation.rotate(ori_img, rotation, center=(self.cam_cx,self.cam_cy), interpolation=cv2.INTER_LINEAR)
                img = data_augmentation.rotate(np.asarray(img), rotation, center=(self.cam_cx,self.cam_cy), interpolation=cv2.INTER_LINEAR)
                depth = data_augmentation.rotate(depth, rotation, center=(self.cam_cx,self.cam_cy), interpolation=cv2.INTER_LINEAR)
                label = data_augmentation.rotate(label, rotation, center=(self.cam_cx,self.cam_cy), interpolation=cv2.INTER_LINEAR)
            
            # test_2 = 1
            # if test_2:
            #     cam_scale = 1.0
            #     camera_intrinsic = [self.width,self.height,self.cam_fx,self.cam_fy,self.cam_cx,self.cam_cy]
            #     ori_img_t = o3d.io.read_image(self.list_rgb[index])
            #     depth_t = o3d.io.read_image(self.list_depth[index])
            #     label_t = o3d.io.read_image(self.list_label[index])
            #     cam = o3d.camera.PinholeCameraIntrinsic()
            #     cam.set_intrinsics(*camera_intrinsic)
            #     target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            #                         ori_img_t, depth_t)
            #     target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            #                 target_rgbd_image, cam)
            #     xyz = np.zeros(np.size(ori_img))
            #     pt2_test = np.reshape(depth / cam_scale, (-1,1))
            #     pt0_test = np.reshape((self.ymap  - self.cam_cx),(-1,1)) * pt2_test / self.cam_fx
            #     pt1_test = np.reshape((self.xmap - self.cam_cy),(-1,1)) * pt2_test / self.cam_fy
            #     #cloud = np.concatenate((pt0_test, pt1_test, pt2_test), axis=1)
            #     pcd = o3d.geometry.PointCloud()
            #     xyz = np.concatenate((pt0_test, pt1_test, pt2_test), axis=1)
            #     pcd.points = o3d.utility.Vector3dVector(xyz/1000)
            #     #o3d.visualization.draw_geometries([pcd])
            #     #o3d.visualization.draw_geometries([target_pcd])
            #     res = np.asarray(target_pcd.points)
            #     color = np.asarray(target_pcd.colors)
            #     K = np.array([[self.cam_fx,0,self.cam_cx],[0,self.cam_fy,self.cam_cy],[0,0,1]])
            #     proj = np.dot(K,res.T)
            #     points_2d = proj.T[:,[0,1]]/proj.T[:,[2]]
            #     x = np.arange(0,640,1)
            #     y = np.arange(0,480,1)
            #     new_img = np.zeros((self.height,self.width,3), np.uint8)
            #     points_2d = np.round(points_2d).astype('int')
            #     new_img[points_2d] = color
            #     #This represents what the network predicts (RED)
            #     if True:
            #         for idx, pt in enumerate(points_2d):
            #             try:
            #                 new_img[pt] = color[idx]
            #             except:
            #                 continue
            #                 #print("outofbond")
            #     print("done")        
                #grid_x, grid_y = np.meshgrid(x, y, indexing='ij')

                #points_2d = np.round(points_2d).astype('int')
                #grid_z1 = griddata(points_2d, color, (grid_x, grid_y), method='linear')
                #scipy.interpolate.RegularGridInterpolator(points_2d, color, method='linear', bounds_error=True, fill_value=0))


                
            #     cloud = cloud / 1000.0
            #test = 0
            # if test:    
            #     rotation = -90
            #     #(self.cam_cy,self.cam_cx)
            #     ori_img = data_augmentation.rotate(ori_img, rotation, center=(self.cam_cx,self.cam_cy), interpolation=cv2.INTER_LINEAR)
            #     img = data_augmentation.rotate(np.asarray(img), rotation, center=(self.cam_cx,self.cam_cy), interpolation=cv2.INTER_LINEAR)
            #     depth = data_augmentation.rotate(depth, rotation, center=(self.cam_cx,self.cam_cy), interpolation=cv2.INTER_LINEAR)
            #     label = data_augmentation.rotate(label, rotation, center=(self.cam_cx,self.cam_cy), interpolation=cv2.INTER_LINEAR)

            # if self.add_noise:
            #     depth = np.reshape(depth, (480,640,1))
            #     depth = np.repeat(depth,3,axis=2) 
                
            #     label = np.reshape(label, (480,640,1))
            #     label = np.repeat(label,3,axis=2) 
                
            #     a = round(random.uniform(0,1))
            #     b = round(random.uniform(0,1))

            #     d = round(random.uniform(0,1))
            #     if d:
            #         ori_img= np.rot90(ori_img, 2)
            #         img = np.rot90(img, 2)
            #         depth = np.rot90(depth, 2)
            #         label = np.rot90(label, 2)

            #     if a:
            #         ori_img = np.fliplr(ori_img)
            #         img = np.fliplr(img)
            #         depth = np.fliplr(depth)
            #         label = np.fliplr(label)
            #     if b: 
            #         ori_img = np.flipud(ori_img)
            #         img = np.flipud(img)
            #         depth = np.flipud(depth)
            #         label = np.flipud(label)
                
            #     c = round(random.uniform(0,1))
            #     if c:
            #         ori_img= np.rot90(ori_img, 2)
            # #         img = np.rot90(img, 2)
            # #         depth = np.rot90(depth, 2)
            # #         label = np.rot90(label, 2)
                
            #     label = label[:,:,0]
            #     depth = depth[:,:,0]
            #traslation_testing = 1
            if self.add_noise: 
                tx, ty = [round(random.uniform(-150,150)), round(random.uniform(-150,150))]
                ori_img = data_augmentation.translate(ori_img, tx, ty, interpolation=cv2.INTER_LINEAR)
                img = data_augmentation.translate(np.asarray(img), tx, ty, interpolation=cv2.INTER_LINEAR)
                depth = data_augmentation.translate(depth, tx, ty, interpolation=cv2.INTER_LINEAR)
                label = data_augmentation.translate(label, tx, ty, interpolation=cv2.INTER_LINEAR)
           
            

            depth = np.round(depth).astype(int)
            obj = self.list_obj[index]
            rank = self.list_rank[index]        

            if obj == 99:
                for i in range(0, len(self.meta[obj][rank])):
                    if self.meta[obj][rank][i]['obj_id'] == 2:
                        meta = self.meta[obj][rank][i]
                        break
            else:
                meta = self.meta[obj][str(rank)][0]
            if self.add_noise:
                label = data_augmentation.random_morphological_transform(label, self.noise_params)
                label = data_augmentation.random_ellipses(label, self.noise_params)
                label = data_augmentation.random_rotation(label, self.noise_params)
                label = data_augmentation.random_add(label, self.noise_params)
                label = data_augmentation.random_cut(label, self.noise_params)
                label = data_augmentation.random_translation(label, self.noise_params)
            labell_dff = label
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            if self.mode == 'eval':
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
            else:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
            # else:  #FIX ME PLEASE!!!
            #     mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
            
            #cv2.imshow("Error bigger than 0.02",label)
            #cv2.waitKey(2000) & 0xFF == ord('q')

            mask = mask_label * mask_depth


            img = np.array(img)[:, :, :3]
            if self.add_noise:
                gaussian = np.round(np.random.normal(0, 5, (img.shape[0],img.shape[1],3)))
                noisy_image = (img + gaussian)
                noisy_image =  np.clip(noisy_image, 0, 255).astype(np.uint8)  
                # apply guassian blur on src image
                blurred_image = cv2.GaussianBlur(noisy_image,(3,3),cv2.BORDER_DEFAULT)
                img = blurred_image

            img = np.array(img)[:, :, :3]
            img = np.transpose(img, (2, 0, 1))
            img_masked = img

            if self.mode == 'eval':
                rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
            else:
                rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
            #rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])
        except:
            self.issue_counter = self.issue_counter + 1
            print("Issue with: {}".format(self.issue_counter))
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc) 
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        #p_img = np.transpose(img_masked, (1, 2, 0))
        #scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))

        target_t = np.array(meta['cam_t_m2c'])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if np.count_nonzero(ori_img[mask_label]==0) > 5:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)
        if np.sum(mask_label == True)<1000:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        Candidate_mask = np.concatenate((ymap_masked,xmap_masked),axis=1)
        choose = np.array([choose])

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        if self.add_noise:
            cloud = np.add(cloud, add_t)


        target_t_re = target_t.reshape(3,1)
        tmp = np.concatenate((target_r,target_t_re),axis=1)
        T = np.concatenate((tmp,[[0,0,0,1]]),axis=0)

        if self.add_noise: 
            px_sh = (-tx_f)*target_t[2]/ self.cam_fx
            py_sh = (-ty_f)*target_t[2]/ self.cam_fy
            T[0,3] = T[0,3] - px_sh
            T[1,3] = T[1,3] -  py_sh

        # if shear_transform:
        #     transf = np.diag([1.,1.,1.,1.])
        #     transf[0:3,0:3] = affine_tf.params
        #     #T = np.dot(transf,T)
            

        if self.add_noise:
                rad_rotations = rotation/360*2*np.pi
                r = np.array(( (np.cos(rad_rotations), -np.sin(rad_rotations)),
                    (np.sin(rad_rotations),  np.cos(rad_rotations)) )).T
                transf = np.diag([1.,1.,1.,1.])
                transf[0:2,0:2] = r
                T = np.dot(transf,T)

        # if test:
        #     K = np.array([[self.cam_fx,0,self.cam_cx],[0,self.cam_fy,self.cam_cy],[0,0,1]])
        #     K_inv = np.array([[self.cam_fy,0,-self.cam_cx*self.cam_fy],[0,self.cam_fx,-self.cam_cy*self.cam_fx],[0,0,self.cam_fx*self.cam_fy]])/(self.cam_fx*self.cam_fy)
        #     rad_rotations = rotation/360*2*np.pi
        #     r = np.array(( (np.cos(rad_rotations), -np.sin(rad_rotations), 0),
        #         (np.sin(rad_rotations),  np.cos(rad_rotations),0),(0,0,1) )).T
        #     #r_new = np.dot(K_inv,np.dot(r,K))
        #     r_test = np.dot(K,r)
        #     transf = np.diag([1.,1.,1.,1.])
        #     transf[0:3,0:3] = r
        #     T = np.dot(transf,T)
        test_depth = 0
        if test_depth:
            K = np.array([[self.cam_fx,0,self.cam_cx],[0,self.cam_fy,self.cam_cy],[0,0,1]])
            K_inv = np.array([[self.cam_fy,0,-self.cam_cx*self.cam_fy],[0,self.cam_fx,-self.cam_cy*self.cam_fx],[0,0,self.cam_fx*self.cam_fy]])/(self.cam_fx*self.cam_fy)
            transf = np.diag([1.,1.,1.,1.])
            transf[2,3] = 0.1
            T = np.dot(transf,T)
            K = np.array([[self.cam_fx,0,self.cam_cx,0],[0,self.cam_fy,self.cam_cy,0],[0,0,1,0],[0,0,0,0]])
            T = T[:3,:]
            test_trasl = np.dot(K,T)
        # if self.add_noise:
        #     if d:
        #         T = np.dot(np.diag([-1,-1,1,1]),T)
        #     if a:
        #         T = np.dot(np.diag([-1,1,1,1]),T)
        #     if b:
        #         T = np.dot(np.diag([1,-1,1,1]),T)
        #     if c:
        #         T = np.dot(np.diag([-1,-1,1,1]),T)
        #     # elif a and b:
        #     #     T = np.dot(np.diag([-1,1,-1,1]),T)
        
        if self.add_noise: 
            px_sh = (-tx)*target_t[2]/ self.cam_fx
            py_sh = (-ty)*target_t[2]/ self.cam_fy
            T[0,3] = T[0,3] - px_sh
            T[1,3] = T[1,3] -  py_sh

        model_points = self.mesh[obj].sample_points_uniformly(self.num)
        target = copy.deepcopy(model_points)
        # target.transform(np.linalg.inv(T))
        target.transform(T)
        target = np.asarray(target.points)
        model_points = np.asarray(model_points.points)
        join = np.concatenate([model_points,target], axis=1)
        np.random.shuffle(join)
        model_points = join[:,:3]
        target  = join[:,3:]

        #target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, add_t)
            out_t =  add_t
        # else:
        #     target = np.add(target, target_t)
        #     out_t = target_t 

        #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
        #for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()
        if self.is_visualized:
            return torch.from_numpy(cloud.astype(np.float32)), \
                torch.LongTensor(choose.astype(np.int32)), \
                self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
                torch.from_numpy(target.astype(np.float32)), \
                torch.from_numpy(model_points.astype(np.float32)), \
                torch.LongTensor([self.objlist.index(obj)]), \
                np.array(ori_img), img_masked, index, Candidate_mask, get_bbox(mask_to_bbox(mask_label)), (target_r,target_t), np.array(labell_dff)
                #    torch.LongTensor([3]),\
        else:
            return torch.from_numpy(cloud.astype(np.float32)), \
                torch.LongTensor(choose.astype(np.int32)), \
                self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
                torch.from_numpy(target.astype(np.float32)), \
                torch.from_numpy(model_points.astype(np.float32)), \
                torch.LongTensor([self.objlist.index(obj)])
                #target_r,target_t_re

               
               
    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    def visualize(self, index):
        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)
        return ori_img



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


# def ply_vtx(path):
#     f = open(path)
#     assert f.readline().strip() == "ply"
#     f.readline()
#     f.readline()
#     N = int(f.readline().split()[-1])
#     while f.readline().strip() != "end_header":
#         continue
#     pts = []
#     for _ in range(N):
#         pts.append(np.float32(f.readline().split()[:3]))
#     return np.array(pts)
