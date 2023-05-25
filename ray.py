import taichi as ti
import taichi.math as tm


@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3

    @ti.func
    def at(self, t: float):
        return self.origin + t * self.direction

    @ti.func
    def normalized_direction(self):
        return self.direction.normalized()

