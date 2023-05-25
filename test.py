import taichi as ti
from ray import Ray
from polygone import Sphere
import taichi.math as tm


ti.init(arch=ti.gpu)

@ti.dataclass
class A:
    a: ti.f32
    b: ti.f32


@ti.dataclass
class B:
    c: A
    d: ti.f32


@ti.data_oriented
class C:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @ti.func
    def hit(self):
        return self.a.a + self.b.c.a + self.b.d

    @ti.func
    def return_dic(self):
        return {'a': 1}

@ti.kernel
def test():
    c = C(1, 2)
    print(c.return_dic())


if __name__ == '__main__':
    test()
