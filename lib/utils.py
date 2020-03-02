import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)
    return l

def save_jpeg(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img/2 + 0.5 
    npimg = img.cpu().detach().numpy()
    if len(npimg.shape)>3:   #Get rid of eventual one dimensions
        npimg = npimg.squeeze()
    if one_channel:
        im = Image.fromarray(npimg)
        im.save("debugFolder/test1.jpeg")
    else:
        npimg = np.transpose(npimg, (1,2,0))
        im = Image.fromarray(npimg, 'RGB')
        im.save("debugFolder/test2.jpeg")
    return npimg

