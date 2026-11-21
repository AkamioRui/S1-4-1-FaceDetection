import copy
import os
import sys
from tqdm import tqdm 
from typing import Callable

from torch import nn
import torch
import torchmetrics
import torchmetrics.classification
print("torch.accelerator.current_accelerator() =",torch.accelerator.current_accelerator())
device = torch.accelerator.current_accelerator()
dtype=torch.float32

torch.set_default_device(device)
torch.set_default_dtype(dtype)



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


if __name__ == '__main__':

    ############### modifiable parameter ###############
    folder_result = os.path.join(__file__,'..','..','result')
    filebase = os.path.basename(__file__).split('.')[0]

    class myDataset(torch.utils.data.IterableDataset):
        def __init__(
                self,
                generator:torch.Generator,
                target_fn:Callable[[float,float],float]
            ):
            
            self.g = generator
            self.t_fn = target_fn
        def __iter__(self):
            while True:
                data = 2*torch.rand((2),generator=self.g)-1
                target = self.t_fn(*data.tolist())
                yield (data,target)
    generator = torch.Generator(device=device).manual_seed(0)
    target_BCE_fn = lambda x,y: x**2 + y**2 <= 0.5**2
    target_exp_fn = lambda x,y: float(target_BCE_fn(x,y))*2-1

    batchSize = 200
    epoch = 500    
    
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

    ############### training ###############
    def train(
        dataLoader:torch.utils.data.DataLoader, #IterableDataset -> (data,target)
        # batchSize,
        # target:torch.Tensor,
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
            for [data,target],i in tqdm(zip(dataLoader,range(epoch))):
                assert isinstance(data,torch.Tensor)
                assert isinstance(target,torch.Tensor)
                
                pred:torch.Tensor = model(data)
                pred = pred.squeeze()
                loss_target = target.to(torch.float32)
                loss:torch.Tensor = loss_fn(pred,loss_target)
                acc:torch.Tensor = acc_fn(pred,target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # store accuracy and parameter changes
                file_acc_fp.write(f'{float(acc.detach())},{float(loss.detach())}\n')
                

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
            print('')
            

    model = MyNN()
    model.to(device=device,dtype=dtype)

    
    
    # TODO model still overfitting, because the training data is only 200 
    ext = 'sgd_exp'
    train(
        dataLoader=torch.utils.data.DataLoader(
            myDataset(generator.clone_state(),target_exp_fn)
            ,batch_size=batchSize
        ),
        model = (_model := copy.deepcopy(model)),
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
        dataLoader=torch.utils.data.DataLoader(
            myDataset(generator.clone_state(),target_BCE_fn)
            ,batch_size=batchSize
        ),
        model = (_model := copy.deepcopy(model)),
        loss_fn = nn.BCEWithLogitsLoss(),
        acc_fn = torchmetrics.classification.BinaryAveragePrecision(),
        epoch = epoch,
        optimizer = torch.optim.SGD(_model.parameters(),lr=1e-2),
        file_acc = os.path.join(folder_result ,f'{filebase}_accuracy_{ext}.txt'),
        file_diff = os.path.join(folder_result ,f'{filebase}_weightDiff_{ext}.txt'),
        file_weight= os.path.join(folder_result ,f'{filebase}_weight_{ext}.bin'),
    )
    ext = 'adam_exp'
    _model = copy.deepcopy(model)
    train(
        dataLoader=torch.utils.data.DataLoader(
            myDataset(generator.clone_state(),target_exp_fn)
            ,batch_size=batchSize
        ),
        model = (_model := copy.deepcopy(model)),
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
        dataLoader=torch.utils.data.DataLoader(
            myDataset(generator.clone_state(),target_BCE_fn)
            ,batch_size=batchSize
        ),
        model = (_model := copy.deepcopy(model)),
        loss_fn = nn.BCEWithLogitsLoss(),
        acc_fn = torchmetrics.classification.BinaryAveragePrecision(),
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