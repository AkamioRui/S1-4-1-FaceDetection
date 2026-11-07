import PIL.Image
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.datasets import WIDERFace
from torchvision.transforms import v2




# generator function --> PIL
dataset = WIDERFace(
    root="myDataset",
    download=True,
    )
# get image, 

img , target = dataset[0]
import PIL.Image as Image
img: Image.Image
img.show()
print("type(img) =",type(img))
print(target)

# # generator function --> PIL
# dataset = WIDERFace(
#     root="myDataset",
#     download=True,
#     transform= v2.Compose(v2.ToImage(),v2.ToDtype())
#     )
# import matplotlib.pyplot as plt
# img,target = dataset[0]
# plt.imshow(img.permute(1,2,0))



""" 
torchvision.dataset --> PIL image using pillow
toImage: PIL -> torch.tensor::torchvision.tv_tensor.image (uint8)
toDtype: uint8 -> float32

image,target = dataset[0]
how do see the image?

how does the model incoorporate target? it has different shape depending on the dataset?


 """



