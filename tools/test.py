import torch
import open3d.ml.torch as ml3d

# +0.5 to move the points to the voxel center
inp_positions = torch.randint(0, 10, [20,3]).to(torch.float32)+0.5
inp_features = torch.randn([20,8])
out_positions = torch.randint(0, 10, [20,3]).to(torch.float32)+0.5

conv = ml3d.layers.SparseConv(in_channels=8, filters=16, kernel_size=[3,3,3])
out_features = conv(inp_features, inp_positions, out_positions, voxel_size=1.0)