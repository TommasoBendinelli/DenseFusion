import pickle
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
opt_dataset_root = "./datasets/linemod/Linemod_preprocessed"
num_points = 500

with open("testdataset_eval.pkl", "wb") as output:
    testdataset = PoseDataset_linemod('eval', num_points, False, opt_dataset_root, 0.0, True)
    pickle.dump(testdataset, output, pickle.HIGHEST_PROTOCOL)

