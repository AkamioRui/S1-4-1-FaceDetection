import os
import sys

from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import xyTest
import torch
from torch import nn

filebase =os.path.basename(__file__).split('_')[0]
ext='_adam_BCE'
file_weigth = os.path.join(__file__,'..','..','result',f'{filebase}_weight{ext}.bin')
# file_weigth = os.path.abspath( os.path.join(__file__,'..','..','result',f'{filebase}_weight{ext}.bin'))


############### input coordinate ###############
max_xy = 2          # x,y [-2,2]
division = 1000
_ = torch.arange(-max_xy,max_xy,2*max_xy/division) 
x = _.unsqueeze(0).repeat([division,1])
y = _.flip(0).unsqueeze(1).repeat([1,division])
xy = torch.stack([x,y],2)

############### recreating the model ###############
model = xyTest.MyNN()
sd = torch.load(file_weigth,weights_only=True)
model.load_state_dict(sd)
# (0): Linear(in_features=2, out_features=5, bias=True)
# (1): ReLU()
# (2): Linear(in_features=5, out_features=3, bias=True)
# (3): ReLU()
# (4): Linear(in_features=3, out_features=1, bias=True)


############### hook + forward  ###############
def forwardhook_saveTo(obj,name):
    def hook(module,args,output):
        obj[name] = output.detach().clone()
    return hook
layer_activation = {
    'prc1':torch.tensor(0),
    'sig1':torch.tensor(0),
    'prc2':torch.tensor(0),
    'sig2':torch.tensor(0),
    'prc3':torch.tensor(0),
}
weight = {
    'prc1':model.block[0]._parameters['weight'],
    # 'sig1':model.block[1]._parameters['weight'],
    'prc2':model.block[2]._parameters['weight'],
    # 'sig2':model.block[3]._parameters['weight'],
    'prc3':model.block[4]._parameters['weight'],
}
h = [
    model.block[0].register_forward_hook(forwardhook_saveTo(layer_activation,'prc1')),
    model.block[1].register_forward_hook(forwardhook_saveTo(layer_activation,'sig1')),
    model.block[2].register_forward_hook(forwardhook_saveTo(layer_activation,'prc2')),
    model.block[3].register_forward_hook(forwardhook_saveTo(layer_activation,'sig2')),
    model.block[4].register_forward_hook(forwardhook_saveTo(layer_activation,'prc3')),
]
model(xy)
[x.remove() for x in h]

############### target circle ###############
circle = x**2 + y**2 
circle = (circle-0.5**2).abs() <= 0.025 


############### rendering pecetron ###############
def render_result(data:torch.Tensor,target:torch.Tensor,extent:tuple[int,int,int,int],subplot:tuple[int,int,int]):
    data = data.detach().to(device='cpu')
    target = target.detach().to(device='cpu')

    plt.subplot(*subplot)
    im = plt.imshow(
        data,
        extent=extent,

        cmap = (cmap:=ListedColormap([
            *[((5-x)/5,0,0) for x in range(5)],
            (1,1,1),
            *[(0,0,(x+1)/5) for x in range(5)],
        ])),
        norm=BoundaryNorm(
            bounds:=[
                *[(5-x)/5*data.min() for x in range(5)] ,
                data.min()/100, data.max()/100, 
                *[(x+1)/5*data.max() for x in range(5)]
            ]
            ,cmap.N
        ),
        
        
    )
    plt.colorbar(im,ax=plt.gca(),ticks=bounds)

    
    plt.imshow(
        target,
        extent= extent,
        cmap= ListedColormap([(0,0,0,0),(0,1,0,1)]),
        norm= BoundaryNorm([0,0.5,1],2,),
    )

    def format_coord(x,y):
        """ 
        extent = (-2, 2, -2, 2) 

        """
        # x,y to data's[idx_x,idx_y]
        N = data.shape[0]
        idx_x = round((x-extent[0])/(extent[1]-extent[0])*(N-1))
        idx_y = round((extent[3]-y)/(extent[3]-extent[2])*(N-1))
        value = float(data[idx_y,idx_x])
        return f'x={x:.2f}, y={x:.2f}\n [{value:.2g}]'
    plt.gca().format_coord = format_coord
    plt.gca().format_cursor_data = lambda data: ''
    return plt.gca()

# plt.figure(
#     'preceptron output',
#     dpi =(dpi := 100),
#     figsize=(1800/dpi,750/dpi),
#     layout='tight',
# )

# for module in ('prc1','prc2','prc3'):
#     output = layer_activation[module]
#     N = output.shape[3]
#     for idx in range(N):
#         render_result(
#             output[:,:,idx],
#             circle,
#             (-max_xy, max_xy,-max_xy, max_xy),
#             (3,N, idx+1)
#         )

############### rendering borders ###############

plt.figure(
    'linear region',
    dpi =(dpi := 100),
    figsize=(1800/dpi,750/dpi),
    layout='tight',
)
# subplot set manually
border = torch.zeros(xy.shape[:2]).to(dtype=torch.bool)
for module,fig_depth in zip(('prc1','prc2','prc3'),(0,1,2)):
    output = layer_activation[module]
    N = output.shape[2]

    w_layer=weight[module]
    assert isinstance(w_layer,torch.nn.Parameter)
    w_layer = w_layer.tolist()

    for idx in range(N):
        render_result(
            output[:,:,idx],
            border.to(dtype=torch.int8),
            extent := (-max_xy, max_xy,-max_xy, max_xy),
            subplot=(3,5, fig_depth*5 + idx+1+fig_depth)
        )
        if module == 'prc1':continue

        
        
        w_str = ', '.join([f'{x:.2f}' for x in w_layer[idx]])
        plt.title(f'{module}_{idx}\nw=[{w_str}]')
    border = border | ( (0.005*output.amin((0,1),True) <= output)  & ( output <=0.005*output.amax((0,1),True)) ).any(dim=2)
        


plt.show()