import copy
import os
import sys
from turtle import back
from typing import Callable

from torch import nn
import torch
print("torch.accelerator.current_accelerator() =",torch.accelerator.current_accelerator())
device = torch.accelerator.current_accelerator()
dtype=torch.float32

torch.set_default_device(device)
torch.set_default_dtype(dtype)

""" 
model objective: { for all (x,y) | 0<=x<=2, 0<=y<=2} --> boolean( y>=(x-1)^2+1 )
"""


class MyNN(nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.block = nn.Sequential(
            # first layer
            nn.Linear(2,5),
            nn.ReLU(),

            # second layer
            nn.Linear(5,3),
            nn.ReLU(),

            # Head layer
            nn.Linear(3,1),
        )
        
    def forward(self,x):
        x = self.block(x)
        return x
model = MyNN()
model.to(device=device,dtype=dtype)





def target_fn(data:torch.Tensor):
    r'''
    data is [[x,y], ...]
    
    ->target[N]: [ -1 or 1, ...]
    '''
    assert data.dim() == 2 and data.shape[1] == 2, f"{data.shape=} =/= (N,2)"

    target = (data[:,0]**2 + data[:,1]**2 ) # f(x,y)
    target = 2*( target < 0.5**2) -1        # threshold {-1,1}
    
    # assert target.dim() == 1, f"{target.shape=} =/= (N)"
    # assert (target.abs() == 1).all(), f"target should have all {{-1,1}}"
    return target 
def loss_exp_fn(pred:torch.Tensor,target:torch.Tensor): # exponential loss (adaboost style)
    r""" 
    target[N]: [ -1 or 1, ...]\
    pred  [N x 1]: any real number, + ment +1, - ment -1

    output[1]
    """
    pred = pred.squeeze()
    assert target.dim() == 1,    f"{target.shape=} =/= (N,)"
    assert (target.abs() == 1).all(),       f"target should have all {{-1,1}}"
    assert pred.dim() == 1,      f"{pred.shape=} =/= (N,)"
    
    output = torch.exp(pred * target * (-1)).mean()
    return output 
def acc_exp_fn(pred:torch.Tensor,target:torch.Tensor):
    r'''
    target[N]: [ -1 or 1, ...]\
    pred  [N x 1]: any real number, + ment +1, - ment -1

    output[1]
    '''
    assert target.dim() == 1,    f"{target.shape=} =/= (N,)"
    assert (target.abs() == 1).all(),       f"target should have all {{-1,1}}"
    assert pred.dim() == 1,      f"{pred.shape=} =/= (N,)"
        
    output = ((pred*target > 0)*1.0).mean()
    return output
def acc_bce_fn(pred:torch.Tensor,target:torch.Tensor,threshold:float):
    r'''
    target[N]: [ -1 or 1, ...]\
    pred  [N x 1]: any real number, + ment +1, - ment -1

    output[1]
    '''
    assert target.dim() == 1,    f"{target.shape=} =/= (N,)"
    assert ( (target-0.5).abs() == 0.5 ).all(),       f"target should have all {{0,1}}"
    assert pred.dim() == 1,      f"{pred.shape=} =/= (N,)"

    pred_class = (nn.Sigmoid()(pred) > threshold)*1.0
    output = ((pred_class*target > 0)*1.0).mean()
    return output

def train(
    data:torch.Tensor,
    # batchSize,
    target:torch.Tensor,
    model:nn.Module,
    loss_fn:Callable[[torch.Tensor,torch.Tensor],torch.Tensor],
    acc_fn:Callable[[torch.Tensor,torch.Tensor],torch.Tensor],
    epoch:int,
    optimizer:torch.optim.Optimizer,
    file_acc:str,
    file_diff:str,
    file_weight:str,
):
    with (
        open(file_acc,'w') as file_acc_fp, 
        open(file_diff,'w') as file_diff_fp,
    ):
        for i in range(epoch):
        # for i in range(1):
            
            pred:torch.Tensor = model(data).squeeze()
            loss:torch.Tensor = loss_fn(pred,target)
            acc:torch.Tensor = acc_fn(pred,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # # DEBUG print
            # print(f"{data.size()=}")
            # print(f"{target.size()=}")
            # print(f"{pred.size()=}")
            # foo = torch.concat([
            #     data,
            #     target.unsqueeze(1),
            #     pred,
            # ],1)
            # print(foo)
            # print(f"{acc=}")
            # print(f"{loss=}")

            # store accuracy and parameter changes
            file_acc_fp.write(f'{float(acc.detach())},{float(loss.detach())}\n')
            print(f'finished {i} epochs')

            if i == 0:
                oldParam = [x.detach().clone() for x in model.parameters()]
                mse = [0 for o in oldParam]
            else: 
                newParam = [x.detach().clone() for x in model.parameters()]
                mse = [float(((n-o)**2).mean()) for o,n in zip(oldParam,newParam)]
                oldParam = newParam
            file_diff_fp.write(','.join([str(x) for x in mse]) + f'\n')


        # store weight
        torch.save(model.state_dict(),file_weight)
        # store every i-th 



# filename generation
folder_result = os.path.join(__file__,'..','..','result')
filebase = os.path.basename(__file__).split('.')[0]
batchSize = 200
data = 2*torch.rand([batchSize,2])-1
epoch = 2000

# TODO BCE need mAP


# doesnt return different weight
_model = copy.deepcopy(model)
ext = 'sgd_exp'
train(
    data = data,
    target = target_fn(data),
    model = _model,
    loss_fn = loss_exp_fn,
    acc_fn = acc_exp_fn,
    epoch = epoch,
    optimizer = torch.optim.SGD(_model.parameters(),lr=1e-2),
    file_acc = os.path.join(folder_result ,f'{filebase}_accuracy_{ext}.txt'),
    file_diff = os.path.join(folder_result ,f'{filebase}_weightDiff_{ext}.txt'),
    file_weight= os.path.join(folder_result ,f'{filebase}_weight_{ext}.bin'),
)

ext = 'sgd_BCE'
_model = copy.deepcopy(model)
train(
    data = data,
    target = (target_fn(data)+1)/2,
    model = _model,
    loss_fn = nn.BCEWithLogitsLoss(),
    acc_fn = lambda pred,target: acc_bce_fn(pred,target,0.9),
    epoch = epoch,
    optimizer = torch.optim.SGD(_model.parameters(),lr=1e-2),
    file_acc = os.path.join(folder_result ,f'{filebase}_accuracy_{ext}.txt'),
    file_diff = os.path.join(folder_result ,f'{filebase}_weightDiff_{ext}.txt'),
    file_weight= os.path.join(folder_result ,f'{filebase}_weight_{ext}.bin'),
)
ext = 'adam_exp'
_model = copy.deepcopy(model)
train(
    data = data,
    target = target_fn(data),
    model = _model,
    loss_fn = loss_exp_fn,
    acc_fn = acc_exp_fn,
    epoch = epoch,
    optimizer = torch.optim.Adam(_model.parameters(),lr=1e-2),
    file_acc = os.path.join(folder_result ,f'{filebase}_accuracy_{ext}.txt'),
    file_diff = os.path.join(folder_result ,f'{filebase}_weightDiff_{ext}.txt'),
    file_weight= os.path.join(folder_result ,f'{filebase}_weight_{ext}.bin'),
)

# (optimizer,error) = (sgd,BCElogit)
ext = 'adam_BCE'
_model = copy.deepcopy(model)
train(
    data = data,
    target = (target_fn(data)+1)/2,
    model = _model,
    loss_fn = nn.BCEWithLogitsLoss(),
    acc_fn = lambda pred,target: acc_bce_fn(pred,target,0.9),
    epoch = epoch,
    optimizer = torch.optim.Adam(_model.parameters(),lr=1e-2),
    file_acc = os.path.join(folder_result ,f'{filebase}_accuracy_{ext}.txt'),
    file_diff = os.path.join(folder_result ,f'{filebase}_weightDiff_{ext}.txt'),
    file_weight= os.path.join(folder_result ,f'{filebase}_weight_{ext}.bin'),
)
     



# how to create my own dataloader
""" old code

# print grad_fn
# # tensor([0.0958], device='cuda:0', grad_fn=<ViewBackward0>)
node = (pred.grad_fn,-1)
def iterGraph(node:tuple[torch.autograd.graph.Node|None,int], depth=0):
    print("  "*depth +f"{node}")
    assert node[0] is not None
    for child in node[0].next_functions:
        iterGraph(child, depth+1)
iterGraph(node)



 """