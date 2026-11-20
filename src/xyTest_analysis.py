import os
import sys

from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import xyTest
import torch

filebase =os.path.basename(__file__).split('_')[0]
ext='_adam_BCE'
file_weigth = os.path.join(__file__,'..','..','result',f'{filebase}_weight{ext}.bin')
# file_weigth = os.path.abspath( os.path.join(__file__,'..','..','result',f'{filebase}_weight{ext}.bin'))

############### recreating the model ###############
model = xyTest.MyNN()
sd = torch.load(file_weigth,weights_only=True)
model.load_state_dict(sd)
# (0): Linear(in_features=2, out_features=5, bias=True)
# (1): ReLU()
# (2): Linear(in_features=5, out_features=3, bias=True)
# (3): ReLU()
# (4): Linear(in_features=3, out_features=1, bias=True)

############### attaching hook to dump ###############
layer_activation = {
    'relu1':torch.tensor(0),
    'relu2':torch.tensor(0)
}

def forwardhook_saveTo(obj,name):
    def hook(module,args,output):
        obj[name] = output.detach().clone()
    return hook

h = [
    model.block[0].register_forward_hook(forwardhook_saveTo(layer_activation,'relu1')),
    model.block[2].register_forward_hook(forwardhook_saveTo(layer_activation,'relu2'))
]



############### forward model ###############
# input coordinate
max_xy = 2          # x,y [-2,2]
division = 1000
_ = torch.arange(-max_xy,max_xy,2*max_xy/division) 
x = _.unsqueeze(0).repeat([division,1])
y = _.flip(0).unsqueeze(1).repeat([1,division])
xy = torch.stack([x,y],2)

# model output
output:torch.Tensor = model(xy)

# target circle
circle = x**2 + y**2 
circle = (circle-0.5**2).abs() <= 0.025 


############### rendering ###############
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

# image setting
plt.figure(
    dpi := 100,
    figsize=(1800/dpi,750/dpi),
    layout='tight',
)
render_result(layer_activation['relu1'][:,:,0],circle,(-max_xy, max_xy,-max_xy, max_xy),(3,5,1))
render_result(layer_activation['relu1'][:,:,1],circle,(-max_xy, max_xy,-max_xy, max_xy),(3,5,2))
render_result(layer_activation['relu1'][:,:,2],circle,(-max_xy, max_xy,-max_xy, max_xy),(3,5,3))
render_result(layer_activation['relu1'][:,:,3],circle,(-max_xy, max_xy,-max_xy, max_xy),(3,5,4))
render_result(layer_activation['relu1'][:,:,4],circle,(-max_xy, max_xy,-max_xy, max_xy),(3,5,5))

render_result(layer_activation['relu2'][:,:,0],circle,(-max_xy, max_xy,-max_xy, max_xy),(3,5,7))
render_result(layer_activation['relu2'][:,:,1],circle,(-max_xy, max_xy,-max_xy, max_xy),(3,5,8))
render_result(layer_activation['relu2'][:,:,2],circle,(-max_xy, max_xy,-max_xy, max_xy),(3,5,9))

render_result(output,circle,(-max_xy, max_xy,-max_xy, max_xy),(3,5,13))



plt.show()