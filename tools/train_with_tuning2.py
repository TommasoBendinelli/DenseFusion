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
import ray
from ray import tune
from ray.tune import track
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler


def test(search_space):
    opt = search_space['opt'] 
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.outf = 'trained_models/linemod'
        opt.log_dir = '/home/labuser/repos/DenseFusion/experiments/logs/linemod'
        opt.repeat_epoch = 20
    elif opt.dataset == 'tommaso':
        opt.num_objects = 1
        opt.outf = '/home/labuser/repos/DenseFusion/trained_models/tommaso'
        opt.log_dir = '/home/labuser/repos/DenseFusion/experiments/logs/tommaso'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return
    
    search_space['num_points'] = int(round(search_space['num_points']))
    estimator = PoseNet(num_points = search_space['num_points'], num_obj = opt.num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points = search_space['num_points'], num_obj = opt.num_objects)
    refiner.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        search_space['lr'] *= opt.lr_rate
        search_space['w'] *= search_space['w_rate']
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=search_space['lr'])
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=search_space['lr'])

    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)
    elif opt.dataset == 'tommaso':
        dataset = Tommaso_poseDataset('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'tommaso':
        test_dataset = Tommaso_poseDataset('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)

        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = dataset.get_sym_list()
    num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), num_points_mesh, opt.sym_list))

    criterion = Loss(num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(num_points_mesh, opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
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
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, search_space['w'], opt.refine_start)
                
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

                #logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                if train_count % (opt.batch_size) == 0:
                    #print(opt.batch_size)
                    if train_count % (opt.batch_size*16) == 0:
                        logger.info('Current Loss: {0}'.format(loss))
                        #logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0
                    
                
                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        #logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        #logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
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
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, search_space['w'], opt.refine_start)
            

            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))

            test_count += 1

        test_dis = test_dis / test_count
        track.log(mean_accuracy=test_dis)
        #logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            search_space['lr'] *= opt.lr_rate
            search_space['w'] *= search_space['w_rate']
            optimizer = optim.Adam(estimator.parameters(), lr=search_space['lr'])

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=search_space['lr'])

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)
            elif opt.dataset == 'tommaso':
                dataset = Tommaso_poseDataset('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'tommaso':
                test_dataset = Tommaso_poseDataset('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)

            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
            
            opt.sym_list = dataset.get_sym_list()
            num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), num_points_mesh, opt.sym_list))

            criterion = Loss(num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(num_points_mesh, opt.sym_list)
def create_opt_settings(config):
    opt = config['opt'] 
    opt.manualSeed = random.randint(1, 10000)
    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.outf = 'trained_models/linemod'
        opt.log_dir = '/home/labuser/repos/DenseFusion/experiments/logs/linemod'
        opt.repeat_epoch = 20
    elif opt.dataset == 'tommaso':
        opt.num_objects = 1
        opt.outf = '/home/labuser/repos/DenseFusion/trained_models/tommaso'
        opt.log_dir = '/home/labuser/repos/DenseFusion/experiments/logs/tommaso'
        opt.repeat_epoch = 20
    else:
        raise('Unknown dataset')        

    opt.refine_start = False
    opt.decay_start = False
    return opt
def create_search_space_settings(config, opt):
    config['num_points'] = int(round(search_space['num_points']))
def get_data_loaders(config,opt):
    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)
    elif opt.dataset == 'tommaso':
        dataset = Tommaso_poseDataset('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'tommaso':
        test_dataset = Tommaso_poseDataset('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    return dataloader, testdataloader
def model(dataset_num_points_mesh,dataset_sym_list,search_space,opt):
    estimator = PoseNet(num_points = search_space['num_points'], num_obj = opt.num_objects)
    refiner = PoseRefineNet(num_points = search_space['num_points'], num_obj = opt.num_objects)
    criterion = Loss(dataset_num_points_mesh, dataset_sym_list)
    criterion_refine = Loss_refine(dataset_num_points_mesh, dataset_sym_list)
    model = estimator, refiner
    losses =  criterion, criterion_refine
    return model, losses

def train(model,losses,optimizer,train_loader,test_loader,opt):
    estimator, refiner = model
    criterion, criterion_refine = losses 

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
    else:
        estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(train_loader, 0):
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
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, search_space['w'], opt.refine_start)
                
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

                #logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                if train_count % (opt.batch_size) == 0:
                    #print(opt.batch_size)
                    if train_count % (opt.batch_size*16) == 0:
                        logger.info('Current Loss: {0}'.format(loss))
                        #logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0
                    
                
                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        #logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        #logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
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
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, search_space['w'], opt.refine_start)
            

            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))

            test_count += 1

        test_dis = test_dis / test_count
        track.log(mean_accuracy=test_dis)
        #logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            search_space['lr'] *= opt.lr_rate
            search_space['w'] *= search_space['w_rate']
            optimizer = optim.Adam(estimator.parameters(), lr=search_space['lr'])

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=search_space['lr'])

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)
            elif opt.dataset == 'tommaso':
                dataset = Tommaso_poseDataset('train', search_space['num_points'], True, opt.dataset_root, search_space['noise_trans'], opt.refine_start)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'tommaso':
                test_dataset = Tommaso_poseDataset('test', search_space['num_points'], False, opt.dataset_root, 0.0, opt.refine_start)

            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
            
            opt.sym_list = dataset.get_sym_list()
            num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), num_points_mesh, opt.sym_list))

            criterion = Loss(num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(num_points_mesh, opt.sym_list)





class TrainableAPITesting():
    def _setup(self, config):
        self.opt = create_opt_settings(config)
        self.search_space = create_search_space_settings(config, opt)
        self.train_loader, self.test_loader = get_data_loaders(opt,search_space)
        self.model, self.losses = model(dataset.get_num_points_mesh(), dataset.get_sym_list(),config,opt)
        self.optimizer = optimizer = optim.Adam(self.estimator.parameters(), lr=search_space['lr'])

    def _train(self):
        train(self.model,self.losses,self.optimizer,self.train_loader,self.test_loader)
        acc = test()
        return {"mean_accuracy": acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'tommaso', help='ycb or linemod or tommaso')
    parser.add_argument('--dataset_root', type=str, default = '/home/labuser/repos/DenseFusion/datasets/tommaso/tommaso_preprocessed', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'' or ''tommaso_preprocessed'')')
    parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
    parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
    #parser.add_argument('--lr', default=0.0001, help='learning rate')
    #parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
    #parser.add_argument('--w', default=0.015, help='learning rate')  #Change me to 0.015
    #parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate') #Change me to 0.3
    parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
    parser.add_argument('--refine_margin', default=0.018, help='margin to start the training of iterative refinement') #Change me to 0.013
    #parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
    parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
    parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
    parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model') #Fix me
    parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
    parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
    opt = parser.parse_args()
    #(10**(np.random.uniform(3,-6)))*
    search_space = {"lr": tune.sample_from(lambda spec: 0.0001), "decay":0.3, "noise_trans": tune.sample_from(lambda spec: 10**(np.random.uniform(1,-1))*0.03), "num_points": tune.sample_from(lambda spec: round(10**(np.random.uniform(1,-1))*500)), "opt":opt}
    search_space2 = {"opt":opt}
    

    BayesianSearhSpace = {"lr":(0.000001,0.0001), "decay": (0.3,0.2), "noise_trans":(0,0.03), "num_points": (500,3000),"w":(0.05,0.0001),"w_rate":(0.5,0)}

    refinement = True
    if refinement == True:
        opt.resume_posenet = "pose_model_22_0.04841270416657975.pth"
        opt.refine_margin = "0.04845"
        search_space2 = {"num_points": 2999.85, "lr": 0, "opt":opt}
        BayesianSearhSpace = {"w":(0.15,0.00015), "w_rate":(0.5,0.01), "decay": (0.3,0.2), "noise_trans":(0,0.03)}


    algo = BayesOptSearch(BayesianSearhSpace, max_concurrent=5, metric= "mean_accuracy",mode="max", utility_kwargs={
            "kind": "ucb",
            "kappa": 2.5,
            "xi": 0.0
        },
        verbose=2)

    scheduler = AsyncHyperBandScheduler(metric="mean_accuracy", mode="max")
    def stopper(trial_id, result):
        return (result["mean_accuracy"] > 100000 or result["training_iteration"] > 50 or np.isnan(result["mean_accuracy"]))

    #tune.run(main,search_space)
    #test(search_space)
    tune.run(test,verbose=2,search_alg=algo ,config=search_space2, name="Experiment_Pose"   , resources_per_trial={
         "cpu": 8,
         "gpu": 1,
     }, num_samples=30,
     stop=stopper)
    #test(search_space)
