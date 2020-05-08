from PIL import Image
import numpy as np
#('datasets/tommaso/tommaso_preprocessed/data/01/rgb/{}.jpg'.format(input_line))
t = np.loadtxt('datasets/tommaso/tommaso_preprocessed/data/01/train.txt')
np.random.shuffle(t)
np.savetxt('datasets/tommaso/tommaso_preprocessed/data/01/train_shuffle.txt',t.astype(int),fmt='%i')
input_file = open('datasets/tommaso/tommaso_preprocessed/data/01/train_shuffle.txt')
read = np.asarray(Image.open('datasets/tommaso/tommaso_preprocessed/data/01/depth/1203.png'))
np.where(read==0, np.mean(read), read)

read2 = np.asarray(Image.open('datasets/linemod/Linemod_preprocessed/data/01/depth/0692.png'))
print("HE")
# if input_line[0] != '4':
#     continue