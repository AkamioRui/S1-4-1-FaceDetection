import PIL.Image
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.datasets import WIDERFace
from torchvision.transforms import v2
import torchvision.tv_tensors




# # generator function --> PIL
# dataset = WIDERFace(
#     root="myDataset",
#     download=True,
#     )
# # get image, 
# img , target = dataset[0]
# import PIL.Image as Image
# img: Image.Image
# img.show()
# print("type(img) =",type(img))
# print(target)

# generator function --> tvtensor
print("before dataset")
dataset = WIDERFace(
    root="myDataset",
    download=True,
    transform= v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)]) 
    # what if scale false?
    )
print("after dataset")
import matplotlib.pyplot as plt
img,target = dataset[0]

import pickle
with open('deleteme','wb') as f:
    pickle.dump([img,target],f)


# plt.imshow(img.permute(1,2,0))
# plt.axis("off")
# plt.show()



""" 
torchvision.dataset --> PIL image using pillow
toImage: PIL -> torch.tensor::torchvision.tv_tensor.image (uint8)
toDtype: uint8 -> float32

image,target = dataset[0]
how do see the image?

how does the model incoorporate target? it has different shape depending on the dataset?


 """



