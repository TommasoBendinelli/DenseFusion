import copy
import numpy as np 
import torch.utils.data
import torchvision.datasets as dset
import matplotlib.pyplot as plt
from PIL import Image 

class Visualizer():
    def __init__(self,dataset):
        self.dataset = dataset

    def RGB(self,idx):
        img = Image.open(self.dataset.list_rgb[idx])
        ori_img = np.array(img)
        return ori_img
    
    def True_Mask(self,idx):
        img = Image.open(self.dataset.list_label[idx])
        true_mask = np.array(img)
        return true_mask


