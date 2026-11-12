from torch import nn
import torch
print("torch.accelerator.current_accelerator() =",torch.accelerator.current_accelerator())

""" 
model objective: { for all (x,y) | 0<=x<=2, 0<=y<=2} --> boolean( y>=(x-1)^2+1 )
"""

# why do I need to create a class?
class MyNN(nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.relu = nn.Sequential(
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
        x = self.relu(x)
        return x

model = MyNN()
model.to(torch.accelerator.current_accelerator())



print(model)
for name, p in model.named_parameters():
    print(f"{name} = {p.size()} = {p}")



# pred = model(x)                # forward: pred = x * gain
# loss = loss_fn(pred, target)
# optimizer.zero_grad()
# loss.backward()                # (b) autograd computes d(loss)/d(gain)
# optimizer.step()               # (c) gain -= lr * gradient

# pred must reference at least the weight and input(2,4), also how does it calculate derivative?