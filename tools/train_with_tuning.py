# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
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
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.tommaso.dataset import PoseDataset as Tommaso_poseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
from ray import tune
from ray.tune import track

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod or tommaso')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'' or ''tommaso_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')  #Change me to 0.015
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate') #Change me to 0.3
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement') #Change me to 0.013
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = 'pose_model_62_0.048668736155996935.pth',  help='resume PoseNet model') #Fix me
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()



# class TrainDenseFusion(tune.Trainable):
#     def _setup(self,config):
#         self.train_loader, 


def process_data(dataset = 'tommaso', num_points = None):


    if dataset == 'ycb':
        dataset = PoseDataset_ycb('train',num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'tommaso':
        dataset = Tommaso_poseDataset('train', num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'tommaso':
        test_dataset = Tommaso_poseDataset('test', num_points, False, opt.dataset_root, 0.0, opt.refine_start)

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    return dataloader, testdataloader, opt.sym_list, opt.num_points_mesh 
    
    return num_objects,num_points,outf,log_dir, repeat_epoch

def model_initialization(num_points):
    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
    estimator.cuda()
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    #testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    optimizer = optim.Adam(estimator.parameters(), lr=config["lr"])
    refiner.cuda()
    return estimator, refiner 

def train_estimator(config):
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    estimator, refiner = model_initialization(num_points = config['num_points'])


    num_objects, num_points,outf,log_dir, repeat_epoch = process_data(dataset="tommaso", num_points = config['num_points'])
    dataloader, testdataloader, opt.sym_list, opt.num_points_mesh  = process_data(dataset = "tommaso")
    
    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
    estimator.cuda()
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    #testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    optimizer = optim.Adam(estimator.parameters(), lr=config["lr"])
    refiner.cuda()
    

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    for epoch in range(opt.nepoch):
        #logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        #logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                try:
                    points, choose, img, target, model_points, idx = data
                except: 
                    continue
                
                points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                    Variable(choose).cuda(), \
                                                                    Variable(img).cuda(), \
                                                                    Variable(target).cuda(), \
                                                                    Variable(model_points).cuda(), \
                                                                    Variable(idx).cuda()                
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
                
                
                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        pred_r, pred_t = pred_r, pred_t
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        dis.backward()
                else:
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                if train_count % (opt.batch_size) == 0:
                    #print(opt.batch_size)
                    if train_count % (opt.batch_size*8) == 0:
                        logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))

            test_count += 1

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= config['lr_decay']
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'tommaso':
                dataset = Tommaso_poseDataset('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'tommaso':
                test_dataset = Tommaso_poseDataset('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)

            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
            
            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

def main():

    # estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    # estimator.cuda()
    # refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    # refiner.cuda()

    # if opt.resume_posenet != '':
    #     estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    # if opt.resume_refinenet != '':
    #     refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
    #     opt.refine_start = True
    #     opt.decay_start = True
    #     opt.lr *= opt.lr_rate
    #     opt.w *= opt.w_rate
    #     opt.batch_size = int(opt.batch_size / opt.iteration)
    #     optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    # else:
    #     opt.refine_start = False
    #     opt.decay_start = False
    #     optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    # if opt.dataset == 'ycb':
    #     dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    # elif opt.dataset == 'linemod':
    #     dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    # elif opt.dataset == 'tommaso':
    #     dataset = Tommaso_poseDataset('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    # if opt.dataset == 'ycb':
    #     test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    # elif opt.dataset == 'linemod':
    #     test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    # elif opt.dataset == 'tommaso':
    #     test_dataset = Tommaso_poseDataset('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)

    # testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    # opt.sym_list = dataset.get_sym_list()
    # opt.num_points_mesh = dataset.get_num_points_mesh()
    

    # print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    # criterion = Loss(opt.num_points_mesh, opt.sym_list)
    # criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    # best_test = np.Inf

    # if opt.start_epoch == 1:
    #     for log in os.listdir(opt.log_dir):
    #         os.remove(os.path.join(opt.log_dir, log))
    # st_time = time.time()

    #search_space = {"lr":tune.sample_from(lambda spec: 10**(-10*np.random.rand())), "decay": tune.sample_from(lambda spec: 10**(-10*np.random.rand()))}

    def get_data_loaders(config):
        if arg.dataset == 'ycb':
            dataset = PoseDataset_ycb('train',config.num_points, True, opt.dataset_root, opt.noise_trans, config.refine_start)
        elif opt.dataset == 'linemod':
            dataset = PoseDataset_linemod('train', config.num_points, True, opt.dataset_root, opt.noise_trans, config.refine_start)
        elif opt.dataset == 'tommaso':
            dataset = Tommaso_poseDataset('train', config.num_points, True, './datasets/tommaso/tommaso_preprocessed', config.noise_trans, config.refine_start)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
        if arg.dataset == 'ycb':
            test_dataset = PoseDataset_ycb('test', config.num_points, False, opt.dataset_root, 0.0, config.refine_start)
        elif arg.dataset == 'linemod':
            test_dataset = PoseDataset_linemod('test', config.num_points, False, opt.dataset_root, 0.0, config.refine_start)
        elif arg.dataset == 'tommaso':
            test_dataset = Tommaso_poseDataset('test', config.num_points, False, './datasets/tommaso/tommaso_preprocessed', 0.0, config.refine_start)

        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

        sym_list = dataset.get_sym_list()
        num_points_mesh = dataset.get_num_points_mesh()
        return dataloader, testdataloader, sym_list, num_points_mesh 
    

    class TrainDenseFusion(tune.Trainable):
        def _setup(self,config):
            self.train_loader, self.self_test_loader, self.sym_list, self.num_points_mesh = get_data_loaders(config)
            self.estimator = PoseNet(num_points = config.num_points, num_obj = config.num_objects).cuda()
            self.refiner = PoseRefineNet(num_points = config.num_points, num_obj = config.num_objects).cuda()
            self.optimizer = optim.Adam(self.estimator.parameters(), lr=config["lr"])

        def _train(self):
            train(self.)



    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    #testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    optimizer = optim.Adam(estimator.parameters(), lr=config["lr"])
    refiner.cuda()        

    class Settings():
        def __init__(self,opt,config):

            #Fixed
            self.dataset = opt.dataset
            if self.dataset == 'ycb':
                self.num_objects = 21 #number of object classes in the dataset
                self.num_points = 1000 #number of points on the input pointcloud
                self.outf = 'trained_models/ycb' #folder to save trained models
                self.log_dir = 'experiments/logs/ycb' #folder to save logs
                self.repeat_epoch = 1 #number of repeat times for one epoch training
            elif self.dataset == 'linemod':
                self.num_objects = 13
                self.num_points = 500
                self.outf = 'trained_models/linemod'
                self.log_dir = 'experiments/logs/linemod'
                self.repeat_epoch = 20
            elif self.dataset == 'tommaso':
                self.num_objects = 1
                self.num_points = 500
                self.outf = '/home/labuser/repos/DenseFusion/trained_models/tommaso'
                self.log_dir = '/home/labuser/repos/DenseFusion/experiments/logs/tommaso'
                self.repeat_epoch = 20
                self.dataset_root = './datasets/tommaso/tommaso_preprocessed'

            if opt.resume_posenet != '':
                estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

            if opt.resume_refinenet != '':
                refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
                opt.refine_start = True
                opt.decay_start = True
                opt.lr *= opt.lr_rate
                opt.w *= opt.w_rate
                opt.batch_size = int(opt.batch_size / opt.iteration)
                optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
            else:
                opt.refine_start = False
                opt.decay_start = False
                optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

            #Argparse
            self.refine_start = opt.refine_start
            self.workers = opt.workers

            #Configurable parameters:
            self.noise_trans = 0 

            self.refine_start = False
            self.decay_start = False
            self.optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)


        def loader(self):
            if self.dataset == 'ycb':
                dataset = PoseDataset_ycb('train',self.num_points, True, self.dataset_root, self.noise_trans, self.refine_start)
            elif self.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', self.num_points, True, self.dataset_root, self.noise_trans, self.refine_start)
            elif self.dataset == 'tommaso':
                dataset = Tommaso_poseDataset('train', self.num_points, True, self.dataset_root, self.noise_trans, self.refine_start)

                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=self.workers)
            if self.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', self.num_points, False, self.dataset_root, 0.0, self.refine_start)
            elif self.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test',self.num_points, False, self.dataset_root, 0.0, self.refine_start)
            elif self.dataset == 'tommaso':
                test_dataset = Tommaso_poseDataset('test', self.num_points, False, self.dataset_root, 0.0, self.refine_start)

            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=self.workers)
            return dataloader, testdataloader

        def restore(self):
            if self.resume_posenet != '':
                estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

            if opt.resume_refinenet != '':
                refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
                opt.refine_start = True
                opt.decay_start = True
                opt.lr *= opt.lr_rate
                opt.w *= opt.w_rate
                opt.batch_size = int(opt.batch_size / opt.iteration)
                optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
        
        # @property
        # def num_objects(self):
        #     return self._num_objects

        # @property
        # def num_points(self):
        #     return self._num_points

        # @property
        # def outf(self):
        #     return self._outf

        # @property
        # def log_dir(self):
        #     return self._log_dir
            
        # @property
        # def repeat_epoch(self):
        #     return self._repeat_epoch
        
    opt = parser.parse_args()
    setting = Settings(opt.dataset)
    search_space = {"lr": 0.0015, "decay":0.3, "noise_trans":0.03, "num_points":500, "args": opt, "setting": setting}
    #test1 = tune.run(train_estimator, config=search_space)
    train_estimator(search_space)


if __name__ == '__main__':
    main()
