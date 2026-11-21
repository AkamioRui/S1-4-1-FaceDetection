import copy
import sys
import time
from tqdm import tqdm
import pandas as pd
import torch.utils.data as D
import torch

# call to create data, """it will stop the program immediately after"""
def print_data():
    def target_f(x,y):
        return x**2+y**2<=0.5**2


    with open(r'E:\tugas\4_1\SpecialTopic\src\deleteme.txt','w') as fd:
        fd.write('x, y, target\n')
        
        input = torch.rand(1000,2)*4-2
        target = target_f(input[:,0],input[:,1]).unsqueeze(1)
        data = torch.concat((input,target),1)

        for datum in data.tolist():
            fd.write(f'{datum[0]}, {datum[1]}, {datum[2]}\n')

    print('finished creating data, exiting...')
    sys.exit()

g = torch.Generator().manual_seed(0)



class MyData(torch.utils.data.IterableDataset):
    def __init__(self,generator):
        self.g = generator # do I need super().__init__()?
    def __iter__(self):
        while True:
            yield torch.rand((1,3),generator=g)*4-2
myDataLoader = D.DataLoader(MyData(g),batch_size=3)

i = 0
for x,_ in zip(myDataLoader,range(1000)): 
    time.sleep(0.01)