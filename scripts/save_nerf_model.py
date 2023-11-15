"""
Given given config.yml, save weights for trained nerf model 

"""
# %% Setup

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import torch

from nerfstudio.utils.eval_utils import eval_setup

# TODO: argparser: config.yml path
#config, pipeline, checkpoint_path, _ = eval_setup(Path('/home/navlab-exxact/NeRF/nerfstudio/outputs/MoonTestScenario/regional-nerfacto/2023-10-13_135547/config.yml'))
#config, pipeline, checkpoint_path, _ = eval_setup(Path('/home/navlab-exxact/NeRF/nerfstudio/outputs/GESMoonRender/regional-nerfacto/2023-10-25_144723/config.yml'))
#config, pipeline, checkpoint_path, _ = eval_setup(Path('/home/navlab-exxact/NeRF/nerfstudio/outputs/GESMoonRender/regional-nerfacto/2023-11-06_145111/config.yml'))
#config, pipeline, checkpoint_path, _ = eval_setup(Path('/home/navlab-exxact/NeRF/nerfstudio/outputs/GESSanJose/regional-nerfacto/2023-11-08_100320/config.yml'))
config, pipeline, checkpoint_path, _ = eval_setup(Path('/home/navlab-exxact/NeRF/nerfstudio/outputs/GESSanJose/regional-nerfacto/2023-10-24_155331/config.yml'))


# TODO: arg: model component options

# TODO: arg: save path

# torch.save(pipeline.model.field.heightcap_net.state_dict(), 'san_jose_heightcap_net.pth')
# print("saved model")

# print(pipeline.model)

# %% --------------------- Sample heights --------------------- %% #

# Random Nx3 values
N = 512
XY_grid = torch.meshgrid(
    torch.linspace(-2, 2, N),
    torch.linspace(-2, 2, N),
)
XY_grid = torch.stack(XY_grid, dim=-1)
positions = XY_grid.reshape(-1, 2)
#positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=-1)

xy = positions[:, :2].detach().cpu().numpy()
x = xy[:,0] 
y = xy[:,1] 
z = pipeline.model.field.positions_to_heights(positions).detach().cpu().numpy().flatten()

# keep = z < -1.5
# x = x[keep]
# y = y[keep]
# z = z[keep]

fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color=z, colorscale='Viridis'))])
fig.update_layout(title='Elevation Model', width=1500, height=800)
fig.update_layout(scene_aspectmode='data')
fig.show()
fig.write_html("ges_moon_heightcap_net.html")


# %% ------------------ Test computing spatial derivatives ------------------ %% #

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

x = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
y = torch.tensor([0.0, 0.1], dtype=torch.float32, requires_grad=True)
positions = torch.stack([x, y], dim=1)
#positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=-1)
#print("positions: ", positions)
# test_pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, requires_grad=True)

heights = pipeline.model.field.positions_to_heights(positions)
#print("heights: ", heights)
print(gradient(heights, x))

# %% --------------------- Plot spatial derivatives --------------------- %% #

# N = 256
# x, y = torch.meshgrid(
#     torch.linspace(-2, 2, N, requires_grad=True),
#     torch.linspace(-2, 2, N, requires_grad=True),
#     indexing='ij'
# )
# print(x.shape, y.shape)
# XY_grid = torch.stack((x, y), dim=-1)
# print(XY_grid.shape)
# positions = XY_grid.reshape(-1, 2)
# print(positions.shape)
# positions = torch.cat([positions, torch.zeros_like(positions[:, :1])], dim=-1)

# heights = pipeline.model.field.positions_to_heights(positions)
# print(heights.shape)

# fig = px.imshow(heights.reshape(N, N).detach().cpu().numpy())

# # fig = px.imshow(gradient(heights, x).detach().cpu().numpy())
# fig.show()

# %% --------------------- Compute derivatives along path --------------------- %% #

x0 = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True)
xf = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)

t = torch.linspace(0, 1, 100)
positions = x0 + t[:, None] * (xf - x0)
heights = pipeline.model.field.positions_to_heights(positions)
print(positions.shape)
print(gradient(heights, positions).shape)
