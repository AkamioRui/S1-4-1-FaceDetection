import pickle

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.tv_tensors

# # pickle test
# with open('deleteme','rb')as f:
#     img,target = pickle.load(f)
    
# img:torchvision.tv_tensors.Image
# img1 = img.permute([1,2,0])

# x,y,w,h = target['bbox'][0]
# img1[x:x+w-1,y:y+h-1,:] = torch.tensor([0,1,0]).view([1,1,3])
print(torch.tensor([0,1,0]))

d2 = torch.tensor(
[
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [
        [11,12,13],
        [14,15,16],
        [17,18,19]
    ]
]
)
# print("d2 =",d2)
print("d2[1,:] =",d2[1,:,:])
size = [-1]
print(f"{size} =",d2.reshape(size))
# plt.imshow()
# plt.show()



# animation

# save a gif

print("done")