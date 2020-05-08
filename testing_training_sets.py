import cv2 
import numpy as np
#('datasets/tommaso/tommaso_preprocessed/data/01/rgb/{}.jpg'.format(input_line))
t = np.loadtxt('datasets/tommaso/tommaso_preprocessed/data/01/train.txt')
np.random.shuffle(t)
np.savetxt('datasets/tommaso/tommaso_preprocessed/data/01/train_shuffle.txt',t.astype(int),fmt='%i')
input_file = open('datasets/tommaso/tommaso_preprocessed/data/01/train.txt')
import random
import data_augmentation
noise_params = {
    
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
    
}
item_count = 0
while 1:
                item_count += 1
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                read = cv2.imread('datasets/tommaso/tommaso_preprocessed/data/01/rgb/{}.jpg'.format(input_line))
                read_mask = cv2.imread('datasets/tommaso/tommaso_preprocessed/data/01/mask/{}.png'.format(input_line))
                read_mask = read_mask[:,:,0]
                read_mask = data_augmentation.random_ellipses(read_mask, noise_params)
                if round(random.uniform(0,1)):
                    read = data_augmentation.rotate(read, 90, center=None, interpolation=cv2.INTER_LINEAR)
                    read_mask = data_augmentation.rotate(read_mask, 90, center=None, interpolation=cv2.INTER_LINEAR)
                
                # read_mask = data_augmentation.random_rotation(read_mask, noise_params)
                # read_mask = data_augmentation.random_add(read_mask, noise_params)
                # read_mask = data_augmentation.random_cut(read_mask, noise_params)
                read_mask = np.reshape(read_mask, (480,640,1))
                read_mask = np.repeat(read_mask,3,axis=2) 
                # read_mask = data_augmentation.random_translation(read_mask, noise_params)
                
                # read_mask = data_augmentation.random_morphological_transform(read_mask, noise_params)
                gray = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
                # Bitwise-OR mask and original image
                cv2.addWeighted(read_mask, 0.4, read, 0.6, 0, read)
                #colored_portion = colored_portion[0:rows, 0:cols]
 
                # Bitwise-OR inverse mask and grayscale image
                # gray_portion = cv2.bitwise_or(gray, gray, mask = mask_inv)
                # gray_portion = np.stack((gray_portion,)*3, axis=-1)
            
                # # Combine the two images
                # output = colored_portion + gray_portion

                cv2.imshow("Masked", read)
                k = 0xFF & cv2.waitKey(100)
                # if input_line[0] != '4':
                #     continue