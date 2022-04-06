import open3d.ml.torch as ml3d
# or import open3d.ml.tf as ml3d
import numpy as np

num_points = 1000000
points = np.random.rand(num_points, 3).astype(np.float32)

data = [
    {
        'name': 'my_point_cloud',
        'points': points,
        'random_colors': np.random.rand(*points.shape).astype(np.float32),
        'int_attr': (points[:,0]*5).astype(np.int32),
    }
]

vis = ml3d.vis.Visualizer()
vis.visualize(data)