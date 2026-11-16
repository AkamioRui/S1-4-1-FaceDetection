
from cProfile import label
import os
import re
import sys

import matplotlib.pyplot as plt

import torch
from torch import nn


def parseData(label=''):
    r'''
    label (xyTest_accuracy_sgd_exp.txt ) = _sgd_exp
    '''
    file_main = os.path.basename(__file__).split('_')[0]
    file_acc = os.path.join(__file__,'..','..','result',f'{file_main}_accuracy{label}.txt')
    file_diff = os.path.join(__file__,'..','..','result',f'{file_main}_weightDiff{label}.txt')

    with open(file_acc,'r') as fp:    
            acc_str = fp.readlines()
            acc = [ [float(c) for c in x.split(',')] for x in acc_str]
    with open(file_diff,'r') as fp:    
        diff_str = fp.readlines()
        diff = [ [float(y) for y in x.split(',')] for x in diff_str]
    
    output = {
        'acc':[x[0] for x in acc],
        'loss':[x[1] for x in acc],
        'weight_diff':diff,
    }
    return output


output_adam_BCE = parseData('_adam_BCE')
output_adam_exp = parseData('_adam_exp')
output_sgd_BCE = parseData('_sgd_BCE')
output_sgd_exp = parseData('_sgd_exp')

dpi = 100
plt.figure(
    dpi=100, 
    figsize=[1800/dpi,750/dpi],
    layout =  'tight'  
)
plt.subplot(1,3,1)
plt.title('acc')
plt.plot(output_adam_BCE['acc'], label = "adam_BCE")
plt.plot(output_adam_exp['acc'],label = "adam_exp" )
plt.plot(output_sgd_BCE['acc'],label = "sgd_BCE" )
plt.plot(output_sgd_exp['acc'],label = "sgd_exp" )
plt.legend()


plt.subplot(2,3,2)
plt.title('loss')
plt.plot(output_adam_BCE['loss'], label = "adam_BCE")
plt.plot(output_sgd_BCE['loss'],label = "sgd_BCE" )
plt.legend()

plt.subplot(2,3,5)
plt.plot(output_adam_exp['loss'],label = "adam_exp" )
plt.plot(output_sgd_exp['loss'],label = "sgd_exp" )
plt.legend()

plt.subplot(4,3,3)
plt.title('d_weight_adam_BCE')
plt.plot(output_adam_BCE['weight_diff'], label = "adam_BCE")
plt.yscale('log')

plt.subplot(4,3,6)
plt.title('d_weight_adam_exp')
plt.plot(output_adam_exp['weight_diff'],label = "adam_exp" )
plt.yscale('log')

plt.subplot(4,3,9)
plt.title('d_weight_sgd_BCE')
plt.plot(output_sgd_BCE['weight_diff'],label = "sgd_BCE" )
plt.yscale('log')

plt.subplot(4,3,12)
plt.title('d_weight_sgd_exp')
plt.plot(output_sgd_exp['weight_diff'],label = "sgd_exp" )
plt.yscale('log')



plt.show()
sys.exit()

output = parseData('_adam_BCE')
plt.subplot(431)
plt.title("acc_adam_BCE")
plt.plot(output['acc'])
plt.subplot(432)
plt.title("loss")
plt.plot(output['loss'])
plt.subplot(433)
plt.title("weight")
plt.yscale('log')
plt.plot(output['weight_diff'])

output = parseData('_adam_exp')
plt.subplot(434)
plt.title("acc_adam_exp")
plt.plot(output['acc'])
plt.subplot(435)
plt.title("loss")
plt.plot(output['loss'])
plt.subplot(436)
plt.title("weight")
plt.yscale('log')
plt.plot(output['weight_diff'])
output = parseData('_sgd_BCE')
plt.subplot(437)
plt.title("acc_sgd_BCE")
plt.plot(output['acc'])
plt.subplot(438)
plt.title("loss")
plt.plot(output['loss'])
plt.subplot(439)
plt.title("weight")
plt.yscale('log')
plt.plot(output['weight_diff'])
output = parseData('_sgd_exp')
plt.subplot(4,3,10)
plt.title("acc_sgd_exp")
plt.plot(output['acc'])
plt.subplot(4,3,11)
plt.title("loss")
plt.plot(output['loss'])
plt.subplot(4,3,12)
plt.title("weight")
plt.yscale('log')
plt.plot(output['weight_diff'])



plt.tight_layout()
plt.show()

