# encoding
- a text file is just a bunch of byte, without metadata
- vscode interpret it based on the encoding at the *footer* (usually utf-8)
- python interpret it by default using `locale.getpreferredencoding(False)`

# python fileIO
- *with open(...) as f:*
    - r+,w+,a+ difference is just how the file is opened, although a+ can only write at the end of the file

# pickle
- serialize an object by creating an instruction on how to construct it. 
```python 
    import pickletool,pickle
    pickle.dumps() #the byte code
    pickletool.dis(pickle.dumps()) # print the disassembled byte code
```
- it roughly include
    - create an empty dict, list, tuple
    - store the elements of above. and once every one is collected, insert it into above
    - if it's a class instance, call the import for the class, then using ```__new__```() and ```__dict__```.update