# import locale

# print("locale.getpreferredencoding(False) =",locale.getpreferredencoding(False))
import pickle

class A:
    def __init__(self):
        self.x = 122
        # self.y = time.localtime()
b = A()

# with open('deleteme','wb') as f:
#     pickle.dump(b,f)

with open('deleteme','rb') as f:
    bb = pickle.load(f)
print(bb.__dict__)

 