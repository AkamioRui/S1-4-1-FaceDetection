# idea
- `pytorch.util.data.dataset` are **abstract class** that can be passed to *pytorch.util.data.dataLoader*
- `pytorch.util.data.dataLoader` are generator function returning data and target each time

- `nn.Module` subclass (here called torch layer) 
    - `torch_layer(x) -> any(but usually torch.Tensor)` takes tensor input x, output a tensor(which has *grad_fn* attr) 
        - nn.Module has overwrite __call__ which inside call forward. so need to define *forward()*
    

# tensor
- var[dim1,dim2,...] = [ [ [] ], [ [] ] ]. outer most are dim1, and the second most are dim2, etc
- `grad_fn`*attr*, is the last node in the *autograd graph*. each node attr: 
    - *name()* 
    - *next_functions()* walks back the graph
    - the rest are node dependent. some store the (input,weight), or parameter reference 
    - whenever a tensor receive a tensor op, `autograd graph` record it, for the purpose of calcualting the gradient
- `detach()` remove *grad_fn*
# tensor op
- class.`einsum`('ab,ac->bc',A,B). A_ab, B_ac, sum along a (the first dimension of both)

- `repeat_interleave` [[1,2],[3,4]] along dim1 -> [[1,2],[1,2],[3,4],[3,4]]
- `repeat` [[1,2],[3,4]] along dim1 -> [[1,2],[3,4],[1,2],[3,4]]

- `unsqueeze`(idx) add a dimension the size of one to the specified idx
- `permute` shuffle dimension
- `reshape` iterate from the highest dimension.
    - (dim1,dim2,dim3), exhaust dim3 first, then dim2, etc

# model creation
```py
class MyModel(nn.Module): 
# nn.Module has overwrite some __...__ function such as __call__ and __init__. it also has other function such as .to()
# nn.Module has modified set attr to now check the value type. if it's a parameter or module, it will be saved in the parameter and module dict, the rest in normal dict, if it's just a tensor it wont be saved

    def __init__(self):
        super().__init__()
        self.myLayer = nn.linear
        
        self.myLayer.register_..._hook(foo)
        # adds a function at that event

        self.myParameter = nn.parameter
        # `weight and bias` attribute stored as *nn.Parameter*
        # `nn.Parameter` is a subclass of torch.Tensor
        
        # clone() make a copy (to actually save it)
        ...

    def forward(self,x):
        # nn.Module.__call__(x) will run self.forward(x) at some point
```
- myModel().to is use to convert data type and change device. 
    - input and weight need to have the same datatype and at the same device

# training
- calculate loss
    - the `nn.Module` for calculating loss is just a module that return a tensor (must be scalar). because to calculate the gradient, all you need is a tensor with it's autograd () 

- calculate gradient

- updating weight


# torchVision

