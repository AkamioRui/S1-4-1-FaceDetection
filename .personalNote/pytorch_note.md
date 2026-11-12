# idea
- *pytorch.util.data.dataset* are **abstract class** that can be passed to *pytorch.util.data.dataLoader*
- *pytorch.util.data.dataLoader*
- nn.Module subclass (here called torch layer) 
    - torch_layer(x) takes tensor input x, output a tensor (which has a grad_fn attr pointing to an obj storing the input)
        - need to defining `forward`(), because nn.Module has overwrite __call__ which inside call forward
    - torch_layer has weight and bias attribute stored as `nn.Parameter`
        - `nn.Parameter` has `grad` attr to be read by `optimizer` and written by`loss`
    - gradient
        - gradient are calculated when ...
        - gradient are calculated by tracking tensor op used

# torch tensor
- var[dim1,dim2,...]
- [ [ [] ], [ [] ] ]. outer most are dim1, and the second most are dim2, etc
- *reshape* iterate from the highest dimension.
    - (dim1,dim2,dim3), exhaust dim3 first, then dim2, etc
# torch model creation
```py
class MyModel(nn.Module): 
# nn.Module has overwrite some __...__ function such as __call__ and __init__. it also has other function such as .to()
# nn.Module has modified set attr to now check the value type. if it's a parameter or module, it will be saved in the parameter and module dict, the rest in normal dict, if it's just a tensor it wont be saved

    def __init__(self):
        super().__init__()
        self.myLayer = nn.linear
        
        self.myLayer.register_..._hook(foo)
        # adds a function at that event
        ...

    def forward(self,x):
        # nn.Module.__call__(x) will run self.forward(x) at some point
```
- myModel().to is use to convert data type and change device. 
    - input and weight need to have the same datatype and at the same device

# torchVision

