# variable
""" 
    casting: int() type()
    :=

    string: 
      "" ''
      + 
      f"a = {var:.2f}"
      '\x11'

    list-tuple:[]
    dict:{
        "foo":"bar"
    }
    set:{}
    var[:-1]
    
    [... for ... in ... if ... ]
    ... if ... else ...
    


    x,y,z = a,b,c
    x,y,z = list
    x,y,*z = list
 """


# control flow
""" 
    if 
        is, and, or, not, in

    for x in []:
    
    while ...:
    
    match ...:
        case ...: 
        case _: 
    
 """

# function
""" 
    def foo():
        sdaf
    def foo(): bar will preserve the context
        n = hhh
        def bar():
            return n
        return bar

    
 """


# class
class a:
    x = 10

a0 = a()
a1 = a()
a1.x = 13
a2 = a()
print("a0.x =",a0.x)
print("a1.x =",a1.x)
print("a2.x =",a2.x)

