
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'tommaso', help='ycb or linemod or tommaso')
    parser.add_argument('--dataset_root', type=str, default = 'datasets/tommaso/tommaso_preprocessed/', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'' or ''tommaso_preprocessed'')')
    parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
    parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
    parser.add_argument('--lr', default=0.0001, help='learning rate')
    parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
    parser.add_argument('--w', default=0.015, help='learning rate')  #Change me to 0.015
    parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate') #Change me to 0.3
    parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
    parser.add_argument('--refine_margin', default=0.05, help='margin to start the training of iterative refinement') #Change me to 0.013
    parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
    parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
    parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
    parser.add_argument('--resume_posenet', type=str, default = 'pose_model_62_0.048668736155996935.pth',  help='resume PoseNet model') #Fix me
    parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
    parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')