# idea
- `pytorch.util.data.dataset` an ABC, task: [idx]->datum
    - *__getitem__(idx)* return (model_input,model_target) one instance of data
    - *__len__()* return total #data
- `pytorch.util.data.dataLoader` takes *dataset* or *IterableDataset*. task: control how data come
    - if you want to pass a tensor as dataset, cast to *pytorch.utils.data.TensorDataset(t1,t2)*. each *next*  return *(t1 slice, t2 slice)*

- `nn.Module` subclass (here called torch layer) 
    - `torch_layer(x) -> any(but usually torch.Tensor)` takes tensor input x, output a tensor(which has *grad_fn* attr) 
        - nn.Module has overwrite __call__ which inside call forward. so need to define *forward()*
    

# tensor
- var[dim1,dim2,...] = [ [ [] ], [ [] ] ]. outer most are dim1, and the second most are dim2, etc
- `tensor storage` the 1D block of raw memory
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
- `stack` (N,) + (N,) ->(N,2)
- `cat` (N,) + (N,) ->(2N,)

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
        
        self.myParameter = nn.parameter
        # `weight and bias` attribute stored as *nn.Parameter*
        # `nn.Parameter` is a subclass of torch.Tensor
        ...

    def forward(self,x):
        # nn.Module.__call__(x) will run self.forward(x) at some point
```
- `to()`.to is use to convert data type and change device. 
    - input and weight need to have the `same datatype and at the same device`
    - it modifies model inplace, but return a clone of a tensor
- `clone()` make a copy 
- `detach()` return a new object without grad, but still refrences the original data
- `register_*_hook`
    - flow: 
        - `forward_pre_hook` -> *forward* -> `forward_hook`
        - `full_backward_pre_hook` -> *backward* -> `full_backward_hook`
    - args:
        - hook format: 
            - `(module,layer_input,kwargs=None)->modified layer_input` for forward_pre_h
            - `(module,layer_input,layer_output,kwargs=None)->modified layer_out` for forward_h
            - the output/input is a reference, in-place changes will modify forward process, even when hook doesnt return
        - `with_kwargs` If False, hook won't get kwargs
        - `prepend` the hook will be run before other hook here, else after
    - return handle for hook, use `handle.remove` to remove
- `amax()` dim=(0,1) then it collapse dim0 and dim1

# training
- calculate loss
    - the `nn.Module` for calculating loss is just a module that return a tensor (must be scalar). because to calculate the gradient, all you need is a tensor with it's autograd () 
- calculate gradient
- updating weight

# save/load
- `torch.save` are just pickle that additionally: 
    - `save the device` the variable is stored (when load, it will default to the same device)
    - `custom persistant_id hook`the resulting file is a zip.
        - *data.pkl* pickle file that store the metadata (data structure), the tensor data are replaced with *token*
        - *data/...* the raw binary that store each tensor data
- `torch.load`
    - `map_location` specify on what device should the data be loaded, default to where the original variable was stored in when calling *torch.save*
    - `weights_only` if true, pickle only call trusted callable, regardless of the opcode.

# torch bool
- only float tensor has NaN. bool tensor have nan converted to 1

