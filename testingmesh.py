import numpy as np
import open3d as o3d
import random
import copy

def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

mesh_test = ply_vtx('datasets/tommaso/tommaso_preprocessed/models/test1.ply')

total_points = np.array(0)
for i in range(1):
    model_points = copy.deepcopy(mesh_test)
    dellist = [j for j in range(0, len(mesh_test))]
    dellist = random.sample(dellist, len(mesh_test) - 500)
    model_points = np.delete(model_points, dellist, axis=0)
    total_points = model_points + total_points
total_points = total_points/1
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(total_points)
pcl.paint_uniform_color([0,0,1])

print(sum(total_points))
mesh_test2 = o3d.io.read_triangle_mesh('datasets/tommaso/tommaso_preprocessed/models/test1.ply')
pcl_2 = o3d.geometry.PointCloud()
total_points2 = np.array(0)
for i in range(1):
    pcl_2.points = mesh_test2.sample_points_uniformly(500).points
    np.random.shuffle(np.asarray(pcl_2.points))
    total_points2 = total_points2 + np.asarray(pcl_2.points)
    
total_points2 = total_points2/1
pcl_2.points = o3d.utility.Vector3dVector(total_points2)
print(sum(np.asarray(total_points2)))
pcl_2.paint_uniform_color([0,1,0])
o3d.visualization.draw_geometries([pcl,pcl_2])


