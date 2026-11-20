layer_activation = {}
def forwardhook_saveTo(obj,name):
    def hook(module,args,output):
        obj[name] = output
    return hook

f = forwardhook_saveTo(layer_activation,'name')
f(2,2,2)
print(f"{layer_activation=}")
