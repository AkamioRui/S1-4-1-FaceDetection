from typing import Any


class A:
    def __call__(self,) -> Any:
        print("hello")
a = A()
a()