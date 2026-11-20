architecture stack for ML in python
*mmdetection* = (from *mmopenlab*) model config (training instruction), training pipeline (image transform)
*mmengine* = (from *mmopenlab*) 
*torchvision* = image dataset, image transform, and model
*pytorch* = somehow send instruction to CUDA. (also has model component, SGD, tensor op )
*CUDA* = API to use NVIDIA GPU 

# python 
- python -X importtime file.py 2>logfile
    if `stdout is not a tty` then python is wrapped by winpty, use python absolute path instead
- python instance is just a dict and a reference to the class, which is just a type instance
- type instance has __name__, __dict__, __parent__, __mro__
- super(__thisclass__,self). will only consider ancestor of  for function definition
    - super(__thisclass__,self).foo will look for implementation of foo from __thisclass__'s ancestor class. this behaviour is defined in C.
    - super(currentClass,self) is the same as super(); 
- `pickle`
    - it store into a .pkl file, one binary file (not a zip archive)
    - `save` when it receive an object:
        - call `persistant_ID`.if return a token, write the opcode for storing a token. if return None, continue;
        - try protocol for native. if fail continue
        - try dispatch table. if fail continue
        - call `__reduce__` if cant then call `object.__reduce_ex__` which will return (callable, (args...)) which is turned into the opcode to be written
    - `load` 
        - call `persistant_load` to load the object ( usally from outside of the .pkl file)
        - normal workflow: follow opcode

# xargs (parallel)
```sh
(process with output) | xargs [opt] "command init_arg"
# process's string output will be read by xargs into arguments
# xargs the run "command init_arg arguments"
# if it took xargs multiple read, then "command init_arg arguments" will be ran multiple times with different arguments each time 
``` 
- **read type**. default = take all argument (seperated by delimiter)
    - *-I `{}`* per line, any `{}` in command is replaced with arguments
    - *-n `_number`* run command with `_number` argument, then repeat the command again with the next `_number` argument
    - *-L `_number`* run command with `_number` line , whitespace before \n will ignore \n, concatinating the next line 
- **delimiter setting**. default = \s
    - *-0* treat process output as raw string (dissable '" processing, \ escaping, _ EOF), so it only delimit at \0
    - *-d `_delimiter`* 
- **output**. 
    - *-P `_number`* run `_number` command in parallel
    - *-t* print each command to &2
    - *-p* as permission each time command run

# awk (sequential)
```sh
awk 'pattern {command} '... file.txt
awk 'NR==1 || $2>max {max=$2; line=$0} END{print line}' file.txt
# read each line, for all pattern and command pair, check if match pattern then run command 
```
- **value**
    - *$0* the whole line
    - *$1,$2,...* token in a line, delimited by 
    - *NF* number of field
    - *NR* current row number 
    - *FS* input field seperator
    - *OFS* output field seperator
- **pattern**
    - */.../* regex
    - *END*
    - normal operator > = || && 
- **command**
    - `var=val`
    - `print`

# sort 
- *-V* sort version number
- *-k2,2* sort based on field 2 to field 2
- *-t,* ',' as delimiter
- *-b* ',' ignore whitespace